import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict 
from model import Model
from collections import defaultdict
from utils import analyze_logits, get_target_label, to_numpy


class RobustnessAnalyzer:
    """
    Class to perform realistic adversarial optimization on 3D scene parameters.

    Args:
        obj_path (str): Path to the object mesh file
        texture_path (str): Path to the texture file
        envmap_paths (List[str]): List of paths to environment maps
        target_class (str): Target class for the attack
        batch_size (int): Batch size for optimization
        targeted (bool): Whether this is a targeted attack
        optimize_kwargs (Dict): Optimization flags for camera, texture, and lighting
        num_iterations (int): Number of optimization iterations
        device (str): Device to run optimization on ('cuda' or 'cpu')
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
        device: str = "cuda"
    ):
        self.obj_path = obj_path
        self.texture_path = texture_path
        self.envmap_paths = envmap_paths
        self.target_class = target_class
        self.batch_size = batch_size
        self.targeted = targeted
        self.device = device

        # Default optimization settings if none provided
        self.nb_clusters = nb_clusters

        # Format needed for the Model instance init
        self.optimize_kwargs = {param: True for param in params_to_optimize}

        # Initialize tracking variables
        self.initial_logits = []
        self.final_logits = []
        self.final_camera_coords = []
        self.camera_positions = []

        # Constants
        self.num_classes = 1000  # ImageNet classes
        self.image_size = 224

        # Setup
        self._setup_model()
        self._setup_target()
        self.loss_fn = nn.CrossEntropyLoss()

    def _setup_model(self) -> None:
        """Initialize the 3D model with given parameters."""
        self.model = Model(
            obj_path=self.obj_path,
            texture_path=self.texture_path,
            envmap_paths=self.envmap_paths,
            optimize_kwargs=self.optimize_kwargs,
            raster_settings={"image_size": self.image_size},
            device=self.device,
            batch_size=self.batch_size,
            nb_clusters=self.nb_clusters
        )
        self.model.train()

    def _setup_target(self) -> None:
        """Setup target tensor for optimization."""
        target = get_target_label(self.target_class)
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

        Args:
            num_runs (int): Number of optimization runs
            num_iterations (int): Number of iterations per run

        Returns:
            Dict containing optimization results
        """
        results = defaultdict(list)

        nb_ims = self.batch_size * (len(self.envmap_paths) or 1)

        for run in range(num_runs):
            try:
                self._setup_model()  # Reset model for new run
                self._create_optimizer(lr=kwargs.get('lr', 1e-1))

                # Initialize lists to track probabilities and losses for this run
                run_avg_probabilities = []
                run_losses = []

                with tqdm(total=num_iterations) as pbar:
                    for i in range(num_iterations):
                        try:
                            self._optimizer_zero_grad()

                            # Add gradient clipping to prevent extreme parameter updates
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.model.scene_params.values() if p.requires_grad],
                                max_norm=1.0
                            )

                            logits = self.model()

                            if i == 0:
                                results['initial_logits'].append(logits.detach().cpu().clone())

                            # Reshape logits and target for loss computation
                            logits_flat = logits.reshape(nb_ims, self.num_classes)
                            target_flat = self.target.reshape(nb_ims, self.num_classes)

                            # Compute loss
                            loss = self.loss_fn(logits_flat, target_flat)
                            loss_  = loss.item()
                            if not self.targeted: # if not targeted, we want to maximize the loss
                                loss *= -1

                            loss.backward()
                            results['grad_norms'].append([torch.norm(param.grad).item() for param in self.model.scene_params.values() if param.grad is not None])  

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

                            pbar.update(1)
                            pbar.set_description(
                                f"Loss: {loss_:.2f} - Class: {class_name:<15}"
                                f" - 'adv acc': {adv_count}/{nb_ims} - Avg Prob: {avg_prob:.4f}"
                                f" - LR: {self.param_groups[0]['lr']:.2e}"
                            )

                        except RuntimeError as iter_err:
                            from ipdb import set_trace; set_trace()
                            print(f"Error in iteration {i}: {iter_err}")
                            continue

                # Store run-level results
                results['loss'].append(run_losses)
                results['avg_probability'].append(run_avg_probabilities)  # true class probability
                results['initial_camera_coords'].append(self.model.camera_coords.detach().cpu())  # Camera position before optimization
                results["initial_scene_params"].append({k: v.detach().cpu().clone() for k, v in self.model.init_scene_params.items()})  # Scene parameters at the end of optimization
                results["final_scene_params"].append({k: v.detach().cpu().clone() for k, v in self.model.scene_params.items()})  # Scene parameters at the end of optimization
                results['final_logits'].append(logits.detach().cpu().clone())  # Final logits after optimization
                results['final_texture'].append(to_numpy(self.model._fill_texture()).copy())  # Final texture after optimization

            except Exception as run_err:
                print(f"Error in run {run}: {run_err}")
                continue

        return results


if __name__ == "__main__":
    robustness_analyzer = RobustnessAnalyzer(
        obj_path="airplane/mesh.obj",
        texture_path="airplane/texture.png",
        envmap_paths=[
            "environments/klippad_dawn_2_k.exr",
            "environments/goegap_road_2k.exr"
        ],
        target_class="space shuttle",
        batch_size=10,
        targeted=True
    )

    # Run optimization
    results = robustness_analyzer.run(num_runs=1)