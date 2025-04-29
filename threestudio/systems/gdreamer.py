from dataclasses import dataclass, field

import copy
import torch
import random
import torch.nn.functional as F

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *
from threestudio.utils.misc import cleanup, get_device

import trimesh
import numpy as np
import mcubes
import os

import pdb


### Helper functions ###


def marching_cubes_with_thickness(
    sdf_grid: np.ndarray,
    res: int,
    scale: float = 3.0,
    center: float = 1.5,
    thickness: float = 0.01,
    level: float = 0.0,
    device=None
) -> trimesh.Trimesh:
    """
    Run Marching Cubes on an SDF grid and create a double-shell mesh with thickness.

    Args:
        sdf_grid (np.ndarray): [res, res, res] SDF values.
        res (int): Resolution of the grid.
        scale (float): Total world size to scale vertices (default 3.0 for [-1.5, 1.5]).
        center (float): Center shift (default 1.5).
        thickness (float): Thickness of the shell (small positive number).
        level (float): Level-set value for outer surface (default 0.0 for SDF).
    
    Returns:
        trimesh.Trimesh: Combined thickened mesh.
    """

    # (1) Extract outer surface
    vertices_outer, faces_outer = mcubes.marching_cubes(sdf_grid, level)
    vertices_outer = vertices_outer / (res - 1) * scale - center  # map to [-center, center]

    # (2) Extract inner surface
    try:
        vertices_inner, faces_inner = mcubes.marching_cubes(sdf_grid, level - thickness)
        vertices_inner = vertices_inner / (res - 1) * scale - center
        # Flip inner faces
        faces_inner = faces_inner[:, [0, 2, 1]]
        valid_inner = vertices_inner.shape[0] > 0 and faces_inner.shape[0] > 0
    except Exception as e:
        print(f"[Warning] Inner surface Marching Cubes failed: {e}")
        valid_inner = False

    # (4) Create trimesh objects
    mesh_outer = trimesh.Trimesh(vertices=vertices_outer, faces=faces_outer, process=False)

    if valid_inner:
        mesh_inner = trimesh.Trimesh(vertices=vertices_inner, faces=faces_inner, process=False)
        solid_mesh = trimesh.util.concatenate([mesh_outer, mesh_inner])
    else:
        solid_mesh = mesh_outer  # Only outer shell

    # Extra robustness:
    solid_mesh.remove_unreferenced_vertices()
    if solid_mesh.faces.shape[0] > 0:
        solid_mesh.remove_degenerate_faces()
        solid_mesh.fix_normals()
    else:
        print(f"[Warning] Solid mesh has no valid faces after construction.")

    return solid_mesh



@threestudio.register("gdreamer-system")
class ObjectDreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        prompt: List[List[str]] = field(default_factory=lambda: [[""],])
        prompt_obj: List[List[str]] = field(default_factory=lambda: [[""],])
        prompt_obj_neg: List[List[str]] = field(default_factory=lambda: [[""],])
        prompt_global: List[List[str]] = field(default_factory=lambda: [[""], ])
        
        prompt_obj_back: List[str] = field(default_factory=lambda: [])
        prompt_obj_side: List[str] = field(default_factory=lambda: [])
        prompt_obj_overhead: List[str] = field(default_factory=lambda: [])
        obj_use_view_dependent: bool = False

        edge_list: List[List[int]] = field(default_factory=lambda: [[],])
       
        obj_use_prompt_debiasing: bool = False
        allow_perp_neg: bool = False
        global_only_steps: int = -1
        new_sdf_loss: bool = False
        enable_global_only_step: bool = True
        whole_graph_start: int = -1
        shuffle_object_order: bool = True
        
        sep_save: bool = False
        test_sep_save: bool = True
        fps: float = 30
        render_edges: List[List[int]] = field(default_factory=lambda: [])

    cfg: Config

    def configure(self):
        super().configure()
        
        num_obj = self.cfg.geometry.num_objects
        self.num_objects = num_obj

        self.render_edges = self.cfg.render_edges[0] if len(self.cfg.render_edges) > 0 and len(self.cfg.render_edges[0]) == 2 else None

        '''1. Prepare guidance model: '''
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        
        self.idx_lst = [i for i in range(num_obj)]
        self.idx_lst.reverse()  # whether object order matters?
        
        '''2. Prepare prompts:'''
        # Global prompt:
        assert self.cfg.prompt_processor.prompt is not None
                    
        prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_utils = prompt_processor()

        self.global_prompt_utils = []
        if len(self.cfg.prompt_global) > 1:
            
            for i, global_prompt_i in enumerate(self.cfg.prompt_global):
                global_prompt_i = ", ".join(global_prompt_i)
                threestudio.info(f"Global prompt for object {i}  [pos]:[{str(global_prompt_i)}]")
                
                tmp_cfg = self.cfg.prompt_processor
                tmp_cfg.prompt = global_prompt_i
                
                _global_processor = threestudio.find(self.cfg.prompt_processor_type)(tmp_cfg)
                global_prompt_utils_i = _global_processor()
                
                self.global_prompt_utils.append(global_prompt_utils_i)
        
        self.edge_list=[]
        if len(self.cfg.edge_list) == num_obj:
            # print("[Edges] for edge rendering:", self.cfg.edge_list)
            self.edge_list = self.cfg.edge_list
        else:
            threestudio.warn("Config 'system.edge_list' for edge rendering is not provided (not necessary for two- and three-object scenes). Use default (cyclic) graph.")
        
        # (1) prepare object prompts: 
        self.obj_use_perp_neg = []
        self.obj_prompt_utils = []
        if num_obj > 1:
            # object prompy:
            for p in self.cfg.prompt_obj:
                print(p)      
            assert (
                len(self.cfg.prompt_obj) == num_obj
            ), f"Given number of object prompts #{len(self.cfg.prompt_obj)} should equals to the object number {num_obj}."
            # negative object prompt:
            if len(self.cfg.prompt_obj_neg) < num_obj:
                for _ in range(len(self.cfg.prompt_obj_neg)):
                    self.obj_use_perp_neg.append(True)
                
                for _ in range(num_obj - len(self.cfg.prompt_obj_neg)):
                    self.cfg.prompt_obj_neg.append("")
                    self.obj_use_perp_neg.append(False)
            else:  # len(self.cfg.prompt_obj_neg) >= num_obj
                self.cfg.prompt_obj_neg = self.cfg.prompt_obj_neg[:num_obj]
                self.obj_use_perp_neg = [True, ] * num_obj
            
            if not self.cfg.allow_perp_neg:
                self.obj_use_perp_neg = [False, ]  * num_obj
            # obj prompt, back view:
            if len(self.cfg.prompt_obj_back) < num_obj:
                for _ in range(num_obj - len(self.cfg.prompt_obj_back)):
                    self.cfg.prompt_obj_back.append("")
            else:
                self.cfg.prompt_obj_back = self.cfg.prompt_obj_back[:num_obj]
            # obj prompt, side view:
            if len(self.cfg.prompt_obj_side) < num_obj:
                for _ in range(num_obj - len(self.cfg.prompt_obj_side)):
                    self.cfg.prompt_obj_side.append("")
            else:
                self.cfg.prompt_obj_side = self.cfg.prompt_obj_side[:num_obj]
            # obj prompt, overhead view:
            if len(self.cfg.prompt_obj_overhead) < num_obj:
                for _ in range(num_obj - len(self.cfg.prompt_obj_overhead)):
                    self.cfg.prompt_obj_overhead.append("")
            else:
                self.cfg.prompt_obj_overhead = self.cfg.prompt_obj_overhead[:num_obj]
            
            # (2) set up a number of object processors:
            for i, (prompt_i, prompt_i_neg, 
                    prompt_i_back, prompt_i_side, prompt_i_overhead,
                    use_perp_neg_i) in enumerate(zip(
                self.cfg.prompt_obj, self.cfg.prompt_obj_neg, 
                self.cfg.prompt_obj_back, self.cfg.prompt_obj_side, self.cfg.prompt_obj_overhead,
                self.obj_use_perp_neg)
            ):  
                try:
                    prompt_i = ", ".join(prompt_i)
                except:
                    print(f"prompt_i \n{prompt_i}")
                prompt_i_neg = ", ".join(prompt_i_neg)
                
                threestudio.info(f"Object {i} prompts [pos]:[{str(prompt_i)}] [neg]:[{str(prompt_i_neg)}]")
                
                tmp_cfg = self.cfg.prompt_processor
                tmp_cfg.prompt = prompt_i
                tmp_cfg.negative_prompt = prompt_i_neg
                
                tmp_cfg.use_view_dependent = self.cfg.obj_use_view_dependent
                tmp_cfg.use_perp_neg = use_perp_neg_i
                
                tmp_cfg.prompt_back = prompt_i_back
                tmp_cfg.prompt_side = prompt_i_side
                tmp_cfg.prompt_overhead = prompt_i_overhead
                
                if not (prompt_i_back is None and prompt_i_side is None and prompt_i_overhead is None):
                    tmp_cfg.use_prompt_debiasing = False  # self.cfg.obj_use_prompt_debiasing

                obj_processor = threestudio.find(self.cfg.prompt_processor_type)(tmp_cfg)
                '''process object prompts:'''
                obj_prompt_utils_i = obj_processor()

                self.obj_prompt_utils.append(obj_prompt_utils_i)
        
    def _single_neg_loss(self, sdf):
        num_negative = torch.sum(sdf < 0, dim=-1)
        return F.relu(num_negative - torch.ones_like(num_negative).to(sdf))
    
    def single_neg_loss(self, sdf_values, min_sdf):
        _, min_indice = torch.min(sdf_values.squeeze(), dim=-1, keepdims=True)
        input = -sdf_values.squeeze() - min_sdf.detach()
        res = torch.relu(input).sum(dim=1, keepdims=True) - torch.relu(torch.gather(input, 1, min_indice))
        loss = res.mean()
        return loss
    
    def argmax_in_forward_softmax_in_backward(
            self,
            x: Float[Tensor, "Nr Di"]
        ) -> Float[Tensor, "*N Di"]:
        fw = F.softmax(1e16 * x, dim=-1)
        fw = torch.nan_to_num(fw, nan=1.0, posinf=1.0, neginf=0.0)
        bw = F.softmax(x, dim=-1)
        return fw.detach() + (bw - bw.detach())  # 0. and 1. with gradients
    
    def intersection_loss(self, sdf_values):
        soft_labels = self.argmax_in_forward_softmax_in_backward(-sdf_values)
        d_inter = F.relu(soft_labels * -sdf_values)
        loss = F.relu(d_inter - sdf_values) ** 2
        return loss.mean()
    
    def forward(self, batch: Dict[str, Any], 
                curr_obj_idx: int = -1) -> Dict[str, Any]:
        
        render_out = self.renderer(**batch, 
                                   curr_obj_idx=curr_obj_idx, global_step=self.true_global_step,
                                   no_local=self.true_global_step<self.cfg.global_only_steps)
        return {
            **render_out,
        }

    def forward_edge(self, batch: Dict[str, Any], 
                     curr_obj_idx: int = -1, 
                     curr_edge: List[int, ] = [-1,],
                     no_local=False) -> Dict[str, Any]:
        
        render_out = self.renderer.forward_edge(
            **batch, 
            curr_obj_idx=curr_obj_idx, global_step=self.true_global_step,
            curr_edge=curr_edge,
            no_local=self.true_global_step<self.cfg.global_only_steps or no_local
        )
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # initialize SDF:
        self.geometry.initialize_shape()
    
    def training_step(self, batch, batch_idx):
        num_obj = self.cfg.geometry.num_objects
                
        # Define current object idx for rendering:
        if num_obj > 1 and self.cfg.enable_global_only_step and self.true_global_step > self.cfg.whole_graph_start:
            curr_obj_idx = self.true_global_step % (num_obj + 1)
        else:
            curr_obj_idx = self.true_global_step % num_obj
        
        if curr_obj_idx < num_obj: # object rendering step
            idx = self.idx_lst[curr_obj_idx]
            if curr_obj_idx == num_obj - 1:  # the last object in current lst
                if self.cfg.shuffle_object_order:
                    random.shuffle(self.idx_lst)
            curr_obj_idx = idx        
        self.log(f"curr_obj_idx:", round(curr_obj_idx,0), prog_bar=True)
        
        loss = 0.0

        """ (1) Global guidance: """ 
        if num_obj > 2 and curr_obj_idx < num_obj:
            # Edge rendering:
            assert len(self.global_prompt_utils) > 0

            # (i) Define edge for each object:
            if len(self.edge_list) == num_obj:
                curr_edge = self.edge_list[curr_obj_idx]
            else:
                if curr_obj_idx == num_obj - 1:
                    curr_edge = [0, curr_obj_idx]
                else:
                    curr_edge = [curr_obj_idx, curr_obj_idx+1]
            
            out = self.forward_edge(batch, curr_obj_idx, curr_edge=curr_edge)
            
            # (ii) Get edge prompt: 
            prompt_utils_edge = self.global_prompt_utils[curr_obj_idx]
            guidance_out = self.guidance(
                out["comp_rgb"], prompt_utils_edge, 
                **batch, rgb_as_latents=False,
            )
        else:  
            # Scene rendering:
            out = self(batch, curr_obj_idx)
                        
            guidance_out = self.guidance(
                out["comp_rgb"], self.prompt_utils, 
                **batch, rgb_as_latents=False,
                scale_up=-1
            )
        
        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                loss_sds = value
                self.log(f"train/{name}", value)
                loss += loss_sds * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        loss_eikonal = (
            (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
        ).mean()
        self.log("train/loss_eikonal", loss_eikonal)
        if self.C(self.cfg.loss.lambda_eikonal) >0:
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        self.log("train/inv_std", out["inv_std"], prog_bar=True)

        if self.cfg.new_sdf_loss:
            loss_sdf = self.single_neg_loss(out["sdf_OBJ"], min_sdf=out["sdf_G"])
        else:
            loss_sdf = self._single_neg_loss(out["sdf"]).mean()

        self.log("train/loss_sdf", loss_sdf)
        if self.C(self.cfg.loss.lambda_sdf) > 0:
            loss += loss_sdf * self.C(self.cfg.loss.lambda_sdf)

        loss_inter = self.intersection_loss(out["sdf_OBJ"])
        self.log("train/loss_inter", loss_inter)
        if self.C(self.cfg.loss.lambda_inter) > 0:
            loss += loss_inter * self.C(self.cfg.loss.lambda_inter)  

        sy = out["soft_Labels"].detach()
        x = out["sdf_OBJ"]
        sx = F.softmax(-x - (-x).max(dim=-1, keepdim=True).values, dim=-1)
        loss_entropy = -F.cross_entropy(sx, sy)
        self.log("train/loss_entropy", loss_entropy)
        if self.C(self.cfg.loss.lambda_entropy) > 0:
            loss += loss_entropy * self.C(self.cfg.loss.lambda_entropy)
        
        if num_obj == 1 or self.true_global_step < self.cfg.global_only_steps or curr_obj_idx == num_obj:  # enable global-only step
            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))
            return {"loss": loss}
        
        """ (2) Object guidance: """ 
        # BP w/ only one object: 
        (
            obj_rgb_i, 
            geo_out_i, 
            prompt_utils_i,
        ) = (
            out["comp_rgb_obj"][curr_obj_idx],
            out["geo_out_obj"][curr_obj_idx], 
            self.obj_prompt_utils[curr_obj_idx],
        )
        
        obj_out = self.guidance(
            obj_rgb_i, 
            prompt_utils_i, 
            **batch, 
            rgb_as_latents=False,
            scale_up=-1,
        )
    
        for name, value in obj_out.items():
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                    
        loss_eikonal_i = (
            (torch.linalg.norm(geo_out_i["sdf_grad_i"], ord=2, dim=-1) - 1.0) ** 2
        ).mean()
        loss += loss_eikonal_i * self.C(self.cfg.loss.lambda_obj_eikonal)

        for i in range(num_obj):
            if i == curr_obj_idx:
                for name, value in obj_out.items():
                    if name.startswith("loss_"):
                        self.log(f"train/{name}_{i}", value)
                
                self.log(f"train/loss_eikonal_{i}", loss_eikonal_i)
            
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        cleanup()
        
        curr_edge = [] 
        out = self(batch)
        
        if self.cfg.sep_save:
            row_g_rgb = [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}_G-rgb.png",
                imgs=row_g_rgb,
                name="validation_step",
                step=self.true_global_step,
                no_compress=True
            )
            row_g_normal = (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}_G-normal.png",
                imgs=row_g_normal,
                name="validation_step",
                step=self.true_global_step,
                no_compress=True
            )
        
        else:
            rows = []
            rows.append( # global visualizations
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ] + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ) + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ] + [
                    {
                        "type": "grayscale",
                        "img": out["comp_mask_fg"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
            )

        if self.num_objects > 1 and self.true_global_step >= self.cfg.global_only_steps:
            for i, (img_obj, opacity_obj) in enumerate(zip(out["comp_rgb_obj"], out["opacity_obj"])):
                
                if len(curr_edge) == 0 or i in curr_edge:

                    if self.cfg.sep_save:
                        row_i_rgb = [
                            {
                                "type": "rgb",
                                "img": img_obj[0],
                                "kwargs": {"data_format": "HWC"},
                            }
                        ]
                        self.save_image_grid(
                            f"it{self.true_global_step}-{batch['index'][0]}_Obj{i}-rgb.png",
                            imgs=row_i_rgb,
                            name="validation_step",
                            step=self.true_global_step,
                            no_compress=True
                        )
                        row_i_normal=(
                            [
                                {
                                    "type": "rgb",
                                    "img": out["comp_normal_obj"][i][0],
                                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                                }
                            ]
                            if "comp_normal_obj" in out
                            else []
                        )
                        self.save_image_grid(
                            f"it{self.true_global_step}-{batch['index'][0]}_Obj{i}-normal.png",
                            imgs=row_i_normal,
                            name="validation_step",
                            step=self.true_global_step,
                            no_compress=True
                        )
                    else:
                        rows.append(  # object visualizations
                            [
                                {
                                    "type": "rgb",
                                    "img": img_obj[0],
                                    "kwargs": {"data_format": "HWC"},
                                }
                            ] + (
                                [
                                    {
                                        "type": "rgb",
                                        "img": out["comp_normal_obj"][i][0],
                                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                                    }
                                ]
                                if "comp_normal_obj" in out
                                else []
                            ) + [
                                {
                                    "type": "grayscale",
                                    "img": opacity_obj[0, :, :, 0],
                                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                                },
                            ] + [
                                {
                                    "type": "grayscale",
                                    "img": out["comp_mask_obj"][i][0, :, :, 0],
                                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                                }
                            ]
                        )
            
        if not self.cfg.sep_save:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}.png",
                imgs=rows,
                name="validation_step",
                step=self.true_global_step,
            )
        cleanup()
    
    def on_validation_epoch_end(self):
        cleanup()
        pass



    @torch.no_grad()
    def export_mesh_from_sdf(self, out, res=128, level=0.0, save_prefix="it{step}-mesh"):
        """
        Export mesh by querying SDF on a dense grid and running Marching Cubes
        """

        # 1. Create a 3D coordinate grid
        lin = torch.linspace(-1.5, 1.5, res, device=get_device())  # Bigger box [-1.5, 1.5]
        grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
        xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # shape [res, res, res, 3]
        xyz_flat = xyz.reshape(-1, 3)

        # 2. Query the model's SDF on this grid
        geo_out = self.geometry(xyz_flat)  # dict_keys(['sdf_OBJ', 'enc_OBJ', 'normal_OBJ', 'shading_normal_OBJ', 'sdf_grad_OBJ', 'features_OBJ', 'soft_Label', 'features_g', 'normal_g', 'shading_normal_g', 'sdf_grad_g'])
        sdf_values = geo_out["sdf_OBJ"].view(res, res, res, -1)
        features_g = geo_out["features_g"].view(res, res, res, -1)  # (res, res, res, C)
        shading_normal_g = geo_out["shading_normal_g"].view(res, res, res, 3)  # (res, res, res, 3)



            # rgb_fg_G: Float[Tensor, "B ... 3"] = self.material(
            #     features=geo_out["features_g"],
            #     viewdirs=t_dirs,
            #     positions=positions,
            #     shading_normal=geo_out["shading_normal_g"],
            #     light_positions=t_light_positions,
            #     **kwargs
            # )    


        
        num_objects = sdf_values.shape[-1]
        output_dir = os.path.join(self.get_save_dir(), "meshes")
        os.makedirs(output_dir, exist_ok=True)
        
        merged_meshes = [] 

        for obj_idx in range(num_objects):
            sdf_grid = sdf_values[..., obj_idx].detach().cpu().numpy()  # [res, res, res] for one object

            # 3. Run Marching Cubes
            # vertices, triangles = mcubes.marching_cubes(sdf_grid, level)
            solid_mesh = marching_cubes_with_thickness(
            sdf_grid=sdf_grid,
            res=res,           # 128 or your resolution
            thickness=0.01,    # can tune this!
            level=0.0          # standard sdf=0 level
            )

            # 4. Assign vertex colors
            vertices = solid_mesh.vertices
            N = vertices.shape[0]

            if N == 0 or solid_mesh.faces.shape[0] == 0:
                print(f"[Warning] Empty mesh for object {obj_idx}, exporting true empty OBJ.")                
                obj_path = os.path.join(output_dir, save_prefix.format(step=self.true_global_step) + f"-obj{obj_idx}.obj")
                solid_mesh.export(obj_path)
                print(f"⚠️ Exported Empty Object {obj_idx}: {obj_path}")

                merged_meshes.append(None)

                continue



            # Map vertices back to grid index to sample features
            v_scaled = ((vertices + 1.5) / 3.0) * (res - 1)
            vi = np.clip(v_scaled[:, 0], 0, res-1).astype(np.int32)
            vj = np.clip(v_scaled[:, 1], 0, res-1).astype(np.int32)
            vk = np.clip(v_scaled[:, 2], 0, res-1).astype(np.int32)

            feature_sample = features_g[vi, vj, vk]
            normal_sample = shading_normal_g[vi, vj, vk]

            # Query material
            feature_sample = torch.tensor(feature_sample, dtype=torch.float32, device=get_device())
            normal_sample = torch.tensor(normal_sample, dtype=torch.float32, device=get_device())
            positions_tensor = torch.tensor(vertices, dtype=torch.float32, device=get_device())

            viewdirs = torch.tensor([0.0, 0.0, 1.0], device=get_device()).expand(positions_tensor.shape)
            light_positions = torch.tensor([[0.0, 10.0, 10.0]], device=get_device()).expand(positions_tensor.shape[0], -1)

            material_outputs = self.material(
                features=feature_sample,
                viewdirs=viewdirs,
                positions=positions_tensor,
                shading_normal=normal_sample,
                light_positions=light_positions
            )

            vertex_colors = material_outputs.detach().cpu().numpy()
            vertex_colors = np.clip(vertex_colors, 0.0, 1.0)

            # Add Alpha if missing
            if vertex_colors.shape[1] == 3:
                alpha_channel = np.ones((vertex_colors.shape[0], 1), dtype=vertex_colors.dtype)
                vertex_colors = np.concatenate([vertex_colors, alpha_channel], axis=1)

            # Attach color to mesh
            solid_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)

            # 5. Export
            obj_path = os.path.join(output_dir, save_prefix.format(step=self.true_global_step) + f"-obj{obj_idx}.obj")
            solid_mesh.export(obj_path)
            print(f"✅ Exported Object {obj_idx}: {obj_path}")

            merged_meshes.append(solid_mesh)

        # 6. Export merged global scene
        valid_meshes = [m for m in merged_meshes if m is not None]

        if len(valid_meshes) > 0:
            global_mesh = trimesh.util.concatenate(valid_meshes)
            global_obj_path = os.path.join(output_dir, save_prefix.format(step=self.true_global_step) + "-G.obj")
            global_mesh.export(global_obj_path)
            # save .glb format too
            global_obj_path_glb = os.path.join(output_dir, save_prefix.format(step=self.true_global_step) + "-G.glb")
            global_mesh.export(global_obj_path_glb)

            print(f"✅ Exported Global Scene: {global_obj_path}, {global_obj_path_glb}")
        else:
            print(f"⚠️ All meshes empty, global scene not exported.")


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        
        if self.render_edges is None:
            out = self(batch)
            # print(f"check out keys: {out.keys()}") # ['sdf_OBJ', 'sdf_G', 'soft_Labels', 'comp_rgb', 'comp_rgb_fg', 'comp_rgb_bg', 'opacity', 'depth', 'comp_mask_fg', 'comp_normal', 'inv_std', 'comp_rgb_obj', 'comp_rgb_fg_obj', 'opacity_obj', 'depth_obj', 'comp_mask_obj', 'comp_normal_obj']
            # shape: out["sdf_OBJ"] [677206, 3]
            # shape: out["sdf_G"] [677206]
            if batch_idx == 0:
                # since the 3D mesh is fixed, we can save it once
                self.export_mesh_from_sdf(out["sdf_G"], res=128, save_prefix=f"it{self.true_global_step}-test")
            no_local = False
            self.postfix = ""

        else:
            curr_idx = 0
            curr_edge = self.render_edges
            self.postfix = f"_{curr_edge[0]}-{curr_edge[1]}"
            no_local = True
            out = self.forward_edge(batch, curr_idx, curr_edge=curr_edge, no_local=True)        
        
        if self.cfg.test_sep_save:
            row_g_rgb = [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}_G-rgb{self.postfix}.png",
                imgs=row_g_rgb,
                name="test_step",
                step=self.true_global_step,
            )
            row_g_normal = (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}_G-normal{self.postfix}.png",
                imgs=row_g_normal,
                name="test_step",
                step=self.true_global_step,
            )
        else:
            rows = []
            rows.append( # global visualizations
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ] + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ) + [
                        {
                            "type": "grayscale",
                            "img": out["opacity"][0, :, :, 0],
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        },
                    ] + [
                        {
                            "type": "grayscale",
                            "img": out["comp_mask_fg"][0, :, :, 0],
                            "kwargs": {"cmap": None, "data_range": (0, 1)},
                        },
                    ]
                )
        
        if self.num_objects > 1 and self.true_global_step >= self.cfg.global_only_steps and not no_local:
            for i, (img_obj, opacity_obj) in enumerate(zip(out["comp_rgb_obj"], out["opacity_obj"])):
                
                if self.cfg.test_sep_save:
                        row_i_rgb = [
                            {
                                "type": "rgb",
                                "img": img_obj[0],
                                "kwargs": {"data_format": "HWC"},
                            }
                        ]
                        self.save_image_grid(
                            f"it{self.true_global_step}-test/{batch['index'][0]}_Obj{i}-rgb.png",
                            imgs=row_i_rgb,
                            name="test_step",
                            step=self.true_global_step,
                            no_compress=True
                        )
                        row_i_normal=(
                            [
                                {
                                    "type": "rgb",
                                    "img": out["comp_normal_obj"][i][0],
                                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                                }
                            ]
                            if "comp_normal_obj" in out
                            else []
                        )
                        self.save_image_grid(
                            f"it{self.true_global_step}-test/{batch['index'][0]}_Obj{i}-normal.png",
                            imgs=row_i_normal,
                            name="test_step",
                            step=self.true_global_step,
                            no_compress=True
                        )
                else:
                    rows.append(  # object visualizations
                        [
                            {
                                "type": "rgb",
                                "img": img_obj[0],
                                "kwargs": {"data_format": "HWC"},
                            }
                        ] + (
                            [
                                {
                                    "type": "rgb",
                                    "img": out["comp_normal_obj"][i][0],
                                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                                }
                            ]
                            if "comp_normal_obj" in out
                            else []
                        ) + [
                            {
                                "type": "grayscale",
                                "img": opacity_obj[0, :, :, 0],
                                "kwargs": {"cmap": None, "data_range": (0, 1)},
                            },
                        ] + [
                            {
                                "type": "grayscale",
                                "img": out["comp_mask_obj"][i][0, :, :, 0],
                                "kwargs": {"cmap": None, "data_range": (0, 1)},
                            },
                        ]
                    )
        
        if not self.cfg.test_sep_save:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                imgs=rows,
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        fps = self.cfg.fps
        
        if self.cfg.test_sep_save:
            
            self.save_img_sequence(
                f"it{self.true_global_step}-test_G-rgb{self.postfix}",
                f"it{self.true_global_step}-test",
                f"(\d+)\_G-rgb{self.postfix}.png",
                save_format="mp4",
                fps=fps,
                name="test",
                step=self.true_global_step,
            )
            self.save_img_sequence(
                f"it{self.true_global_step}-test_G-normal{self.postfix}",
                f"it{self.true_global_step}-test",
                f"(\d+)\_G-normal{self.postfix}.png",
                save_format="mp4",
                fps=fps,
                name="test",
                step=self.true_global_step,
            )
            
            for i in range(self.num_objects):
                self.save_img_sequence(
                    f"it{self.true_global_step}-test_Obj{i}-rgb",
                    f"it{self.true_global_step}-test",
                    f"(\d+)\_Obj{i}-rgb.png",
                    save_format="mp4",
                    fps=fps,
                    name="test",
                    step=self.true_global_step,
                )
                self.save_img_sequence(
                    f"it{self.true_global_step}-test_Obj{i}-normal",
                    f"it{self.true_global_step}-test",
                    f"(\d+)\_Obj{i}-normal.png",
                    save_format="mp4",
                    fps=fps,
                    name="test",
                    step=self.true_global_step,
                )
        
        else:
            self.save_img_sequence(
                f"it{self.true_global_step}-test",
                f"it{self.true_global_step}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=fps,
                name="test",
                step=self.true_global_step,
            )
