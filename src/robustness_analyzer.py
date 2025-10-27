import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import List, Dict, Optional, Tuple
from model import Model
from collections import defaultdict
from utils import analyze_logits, get_target_label, to_numpy


class RobustnessAnalyzer:
    """
    Class to perform realistic adversarial optimization on 3D scene parameters.

    Args:
        obj_path (str): Path to the object mesh file
        texture_path (str): Path to the texture file (or None to use materials from MTL)
        envmap_paths (List[str]): List of paths to environment maps
        target_class (str): Target class for the attack
        batch_size (int): Batch size for optimization
        targeted (bool): Whether this is a targeted attack
        optimize_kwargs (Dict): Optimization flags for camera, texture, and lighting
        num_iterations (int): Number of optimization iterations
        device (str): Device to run optimization on ('cuda' or 'cpu')
        raster_settings (Dict, optional): Dictionary of rasterization settings. Defaults to {"image_size": 224}.
        model_name (str): Name of the classification model to use
    """

    def __init__(
        self,
        obj_path: str,
        texture_path: str,
        envmap_paths: List[str],
        target_class: str,
        params_to_optimize: List[str],
        batch_size: int = 10,
        targeted: bool = True,
        nb_clusters: int = 4,
        positive_z: bool = True,
        device: str = "cuda",
        raster_settings: Optional[Dict] = {},
        model_name: str = "vit-large-patch16-224",
        custom_weights_path: Optional[str] = None,
        min_max_proportion: Tuple[float, float] = (0.3, 0.8)
    ):
        self.obj_path = obj_path
        self.texture_path = texture_path
        self.envmap_paths = envmap_paths
        self.target_class = target_class
        self.batch_size = batch_size
        self.targeted = targeted
        self.device = device
        self.positive_z = positive_z
        self.model_name = model_name
        self.custom_weights_path = custom_weights_path
        self.min_max_proportion = min_max_proportion

        # Default optimization settings if none provided
        self.nb_clusters = nb_clusters

        # Format needed for the Model instance init
        self.optimize_kwargs = {param: True for param in params_to_optimize}

        # Initialize tracking variables
        self.initial_logits = []
        self.final_logits = []
        self.final_camera_coords = []
        self.camera_positions = []

        # Setup
        self.raster_settings = raster_settings
        self._setup_model()

        # Get number of classes from the loaded model
        self.num_classes = self._get_num_classes_from_model()
        print(f"📊 Model has {self.num_classes} output classes")

        self._setup_target()
        self.loss_fn = nn.CrossEntropyLoss()

        # Add state tracking for intermediate results
        self._current_results = defaultdict(list)
        self._current_run = 0
        self._current_iteration = 0

    def _setup_model(self) -> None:
        """Initialize the 3D model with given parameters."""
        self.model = Model(
            obj_path=self.obj_path,
            texture_path=self.texture_path,
            envmap_paths=self.envmap_paths,
            optimize_kwargs=self.optimize_kwargs,
            min_max_proportion=self.min_max_proportion,
            raster_settings=self.raster_settings,
            device=self.device,
            batch_size=self.batch_size,
            nb_clusters=self.nb_clusters,
            positive_z=self.positive_z,
            model_name=self.model_name,
            custom_weights_path=self.custom_weights_path
        )
        self.model.eval()

    def _get_num_classes_from_model(self) -> int:
        """Get the number of output classes from the loaded model."""
        ml_model = self.model.ml_model

        # Check different possible classifier attribute names
        if hasattr(ml_model, 'classifier'):
            classifier = ml_model.classifier
            if hasattr(classifier, 'out_features'):
                return classifier.out_features
            elif hasattr(classifier, 'weight'):
                return classifier.weight.shape[0]
            elif isinstance(classifier, nn.Sequential):
                # For ResNet-style sequential classifiers
                for layer in reversed(classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.out_features
        elif hasattr(ml_model, 'head'):
            head = ml_model.head
            if hasattr(head, 'out_features'):
                return head.out_features
            elif hasattr(head, 'weight'):
                return head.weight.shape[0]

        # Default to 1000 if we can't determine
        print("⚠️  Could not determine number of classes from model, defaulting to 1000")
        return 1000

    def _setup_target(self) -> None:
        """Setup target tensor for optimization."""
        target = get_target_label(self.target_class, num_classes=self.num_classes, device=self.device)
        self.target = target.expand(
            self.batch_size,
            len(self.envmap_paths) or 1,
            -1
        ).to(self.device)

    def _create_optimizer(self, lr: float) -> None:
        """
        Initialize learning rates for each parameter group with proper gradient tracking.
        """
        self.param_groups = []
        for name, param in self.model.scene_params.items():
            if param.requires_grad:
                # Ensure the parameter is a leaf tensor
                if not param.is_leaf:
                    param = param.detach().requires_grad_(True)

                self.param_groups.append({
                    'params': [param],
                    'lr': lr,
                    'name': name
                })

    def _optimizer_zero_grad(self) -> None:
        """
        Zero out gradients for all parameters with proper handling.
        """
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()

    def _optimizer_step(self) -> None:
        """
        Perform SGD update on parameters with gradient clipping.
        """
        # Update each parameter group
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    with torch.no_grad():
                        param.grad.clamp_(-5, 5)
                        param.data -= group['lr'] * param.grad

    def run(self, num_runs: int = 1, num_iterations: int = 100, **kwargs) -> Dict[str, List[torch.Tensor]]:
        """
        Run the optimization process with robust error handling.
        Results are stored in self._current_results and updated continuously.
        """
        self.model.reset_cache() # reset cache to remove cached mesh and texture from previous runs
        self._current_results = defaultdict(list)

        # Get progress callback if provided
        progress_callback = kwargs.get('progress_callback', None)

        with tqdm(range(num_runs), desc="Initializing", mininterval=0.1) as pbar:
            for run in pbar:
                self._current_run = run

                # Update Streamlit progress if callback provided
                if progress_callback:
                    progress_callback(run, num_runs)

                try:
                    self._setup_model()
                    self._create_optimizer(lr=kwargs.get('lr', 1e-1))

                    # Initialize lists to track probabilities and losses for this run
                    run_avg_probabilities = []
                    run_losses = []

                    assert num_iterations > 0, "num_iterations must be greater than 0"
                    if num_iterations == 0:
                        logits = self.model(return_render=False, with_grad=False).detach().cpu().clone()
                        self._current_results['initial_logits'].append(logits)
                        self._current_results['final_logits'].append(logits)
                        pbar.set_description(f"Run {run + 1}/{num_runs} [Skipped: 0 iterations]")
                        continue  # Skip the optimization loop

                    for i in range(num_iterations):
                        self._current_iteration = i

                        # Update progress with iteration info
                        if progress_callback:
                            progress_callback(run, num_runs, i, num_iterations)

                        try:
                            self._optimizer_zero_grad()
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.model.scene_params.values() if p.requires_grad],
                                max_norm=1.0
                            )

                            logits = self.model(return_render=False, with_grad=True)

                            if i == 0:
                                self._current_results['initial_logits'].append(logits.detach().cpu().clone())

                            # Reshape logits and target for loss computation
                            nb_ims = self.batch_size * (len(self.envmap_paths) or 1)
                            logits_flat = logits.reshape(nb_ims, self.num_classes)
                            target_flat = self.target.reshape(nb_ims, self.num_classes)

                            # Compute loss
                            loss = self.loss_fn(logits_flat, target_flat)
                            loss_  = loss.item()
                            if not self.targeted: # if not targeted, we want to maximize the loss
                                loss *= -1

                            loss.backward()

                            # Store GPU memory stats in GB
                            if torch.cuda.is_available():
                                self._current_results["gpu_memory"].append(torch.cuda.memory_allocated() / (1024 ** 3))
                                self._current_results["gpu_memory_reserved"].append(torch.cuda.memory_reserved() / (1024 ** 3))

                            self._current_results['grad_norms'].append(
                                [torch.norm(param.grad).item() for param in self.model.scene_params.values()
                                 if param.grad is not None]
                            )

                            for param in self.model.scene_params.values():
                                if param.grad is not None:
                                    param.grad.clamp_(-1, 1)

                            self._optimizer_step()

                            # Analyze logits
                            class_name, class_count, avg_prob = analyze_logits(
                                logits_flat, self.target_class
                            )

                            # Store current iteration's loss and average probability
                            run_losses.append(loss_)
                            run_avg_probabilities.append(avg_prob)

                            # Calculate how many images are correctly classified according to the attack
                            adv_count = nb_ims - class_count if not self.targeted else class_count

                            # Update progress bar description with iteration info
                            pbar.set_description(
                                f"Run {run + 1}/{num_runs} [Iter {i + 1}/{num_iterations}] - "
                                f"Loss: {loss_:.2f} - Class: {class_name:<15} - "
                                f"Adv Acc: {adv_count}/{nb_ims} - Prob: {avg_prob:.4f} - "
                                f"Model: {self.model_name}"
                            )

                            # Store intermediate results in class state
                            self._current_results['loss'].append(loss_)
                            self._current_results['avg_probability'].append(avg_prob)

                        except RuntimeError as iter_err:
                            torch.cuda.empty_cache()
                            # handle OOM by reducing batch size and reinitializing the model
                            if "out of memory" in str(iter_err).lower():
                                # if batch size is 1, we cannot reduce further
                                if self.batch_size == 1:
                                    raise Exception("Batch size is 1, cannot reduce further, reduce image size or number of environments")
                                self.batch_size = self.batch_size // 2
                                pbar.write(f"GPU OOM error in iteration {i}, reducing batch size to {self.batch_size} and retrying...")
                                self._setup_model()  # Reinitialize with new batch size
                                self._setup_target()  # Update the target tensor with new batch size
                                self.model.reset_cache() # remove cached mesh and texture to adjust to new batch size
                            else:
                                pbar.write(f"Error in iteration {i}: {iter_err}")
                            continue

                    # Store run-level results
                    self._current_results['loss'].append(run_losses)
                    self._current_results['avg_probability'].append(run_avg_probabilities)  # true class probability
                    self._current_results['initial_camera_coords'].append(self.model.camera_coords.detach().cpu())  # Camera position before optimization
                    self._current_results["initial_scene_params"].append({k: v.detach().cpu().clone() for k, v in self.model.init_scene_params.items()})  # Scene parameters at the end of optimization
                    self._current_results["final_scene_params"].append({k: v.detach().cpu().clone() for k, v in self.model.scene_params.items()})  # Scene parameters at the end of optimization
                    self._current_results['final_logits'].append(logits.detach().cpu().clone())  # Final logits after optimization

                    # Only store the final texture if texture optimization was enabled
                    if self.optimize_kwargs.get("texture", False):
                        self._current_results['final_texture'].append(to_numpy(self.model._fill_texture()).copy())

                except Exception as run_err:
                    raise run_err
                    pbar.write(f"Error in run {run}: {run_err}")
                    continue

                # free gpu memory
                del logits, loss, logits_flat, target_flat
                torch.cuda.empty_cache()

        return self._current_results


    def get_current_results(self) -> Dict[str, List[torch.Tensor]]:
        """Get the latest results, even if optimization has failed."""
        return self._current_results

    def save_results(self, file_path: str) -> None:
        """
        Serialize and save the current results to a file.

        Args:
            file_path (str): Path where the results will be saved
        """
        # Make sure all tensors are detached and moved to CPU for proper serialization
        serializable_results = {}

        for key, values in self._current_results.items():
            if isinstance(values, list):
                if len(values) > 0 and isinstance(values[0], dict):
                    # Handle dictionaries of tensors (like scene_params)
                    serializable_results[key] = [
                        {k: v.detach().cpu() if torch.is_tensor(v) else v
                         for k, v in item.items()}
                        for item in values
                    ]
                elif len(values) > 0 and torch.is_tensor(values[0]):
                    # Handle lists of tensors
                    serializable_results[key] = [v.detach().cpu() for v in values]
                else:
                    # Handle regular lists
                    serializable_results[key] = values
            elif torch.is_tensor(values):
                # Handle tensor values
                serializable_results[key] = values.detach().cpu()
            else:
                # Handle other types
                serializable_results[key] = values

        # Save the serialized results
        torch.save(serializable_results, file_path)
        print(f"Results saved to {file_path}")

    def load_results(self, file_path: str) -> Dict:
        """
        Load saved results from a file.

        Args:
            file_path (str): Path to the saved results file

        Returns:
            Dict: The loaded results
        """
        try:
            loaded_results = torch.load(file_path)
            self._current_results = defaultdict(list)

            # Copy loaded results to the current_results
            for key, values in loaded_results.items():
                self._current_results[key] = values

            print(f"Results loaded from {file_path}")
            return dict(self._current_results)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}


if __name__ == "__main__":
    import random, glob, os

    data_dir = "../data"

    # Envmap(s) we're using during optimization (one or more)
    max_envs = 5 # -1 or None for all
    envmap_paths = glob(os.path.join("../data", "environments_road/*"))
    random.shuffle(envmap_paths)
    envmap_paths = envmap_paths[:max_envs]
    print("#environments used:", len(envmap_paths))

    true_class = 'tank, army tank, armored combat vehicle, armoured combat vehicle'
    target_class = true_class

    object_dir = "leopard_tank"

    params_to_optimize = ["camera"]

    raster_settings = {
        "image_size": 448, # image resolution (image_size, image_size, 3)
        "bin_size": 32,  # Controls spatial partitioning for rasterization - larger values use less memory but may be slower
        "max_faces_per_bin": 100_000,  # Maximum faces per spatial bin - increase for complex meshes, decrease to save memory
    }

    kwargs = {
        "obj_path": os.path.join(data_dir, object_dir, "leopard_decimated.obj"),
        "texture_path": None,
        "envmap_paths": envmap_paths,
        "target_class": target_class,
        "batch_size": 2,  # How many different viewpoints we're optimizing in parallel (* #Environments)
        "params_to_optimize": params_to_optimize,
        "targeted": target_class!=true_class,
        "positive_z": True, # constraints camera z>0 (positive elevation)
        "raster_settings": raster_settings,
        "model_name": "vit-large-patch16-224"
    }

    robust_analyzer = RobustnessAnalyzer(**kwargs)

    ##### Run optimization: This will generate num_runs*batch_size viewpoints
    num_runs = 10
    num_iterations = 1
    results_camera = robust_analyzer.run(num_runs, num_iterations, lr=5e-3)
