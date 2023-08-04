from models.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = "osmesa"

import torch
from visualize.simplify_loc2rot import joints2smpl
import pyrender
import matplotlib.pyplot as plt

import io
import imageio
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
import math
# import ffmpeg
from PIL import Image

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def render(motions, outdir='test_vis', device_id=0, name=None, pred=True):
    # print(motions.shape)
    frames, njoints, nfeats = motions.shape
    print(frames,njoints,nfeats)
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    if (not os.path.exists(outdir + name+'_pred.pt') and pred) or (not os.path.exists(outdir + name+'_gt.pt') and not pred): 
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone().detach(), mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        vertstrans=True)
# modified here
        # if pred:
        #     torch.save(vertices, outdir + name+'_pred.pt')
        # else:
        #     torch.save(vertices, outdir + name+'_gt.pt')
    else:
        if pred:
            vertices = torch.load(outdir + name+'_pred.pt')
        else:
            vertices = torch.load(outdir + name+'_gt.pt')
    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
    print (vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5


    out_list = []
    
    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    vid = []
    for i in range(frames):
        if i % 20 == 0:
            print(i,"frames rendered")

        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        base_color = (0.11, 0.53, 0.8, 0.5)
        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha 
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )


        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)

        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

        c = np.pi / 2

        scene.add(polygon_render, pose=np.array([[ 1, 0, 0, 0],

        [ 0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],

        [ 0, np.sin(c), np.cos(c), 0],

        [ 0, 0, 0, 1]]))

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())


        c = -np.pi / 6

        scene.add(camera, pose=[[ 1, 0, 0, (minx+maxx).cpu().numpy()/2],

                                [ 0, np.cos(c), -np.sin(c), 1.5],

                                [ 0, np.sin(c), np.cos(c), max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())],

                                [ 0, 0, 0, 1]
                                ])
        
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        vid.append(color)

        r.delete()

    out = np.stack(vid, axis=0)
    if pred:
        # imageio.mimsave(outdir + name+'_pred.gif', out, fps=20)
        imageio.mimsave(outdir + name+'_pred.gif', out, fps=20)
    else:
        # imageio.mimsave(outdir + name+'_gt.gif', out, fps=20)
        imageio.mimsave(outdir + name+'_gt.gif', out, fps=20)
    




if __name__ == "__main__":
    with torch.cuda.device(7):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--filedir", type=str, default=None, help='motion npy file dir')
        parser.add_argument('--motion-list', default=None, nargs="+", type=str, help="motion name list")
        args = parser.parse_args()

        filename_list = args.motion_list
        filedir = args.filedir
        
        for filename in filename_list:


            # Load the data from the npy file with allow_pickle=True
            motions = np.load(filedir + filename+'_pred.npy', allow_pickle=True)

            # Check the type and dtype of the loaded object
            if isinstance(motions, np.ndarray) and motions.dtype == object:
                print("MDM encountered. The npy file contains a dictionary of arrays.")
                # motions = np.array(motions.item().values())
                motions = motions.item()
                poses = motions['full_pose']
                print(motions['smpl_poses'].shape, motions['smpl_trans'].shape, motions['full_pose'].shape)
                # np.savetxt('array_data.csv', motions['smpl_trans'], delimiter=',', fmt='%f')
                # smpl_poses = motions['smpl_poses'].reshape(900, 24, 3)
                # smpl_trans = motions['smpl_trans'].reshape(900, 1, 3)
                # motions = np.concatenate((smpl_poses, smpl_trans, motions['full_pose']), axis=1)
                # motions = np.broadcast_to(smpl_trans, (900, 24, 3))
                motions = motions['full_pose']
                motions = motions[np.newaxis, :]
                print(motions.shape)
                # motions = motions.item()['motion']
                print("transpose shaper called")
                # motions = np.transpose(motions,axes=(0,3,1,2))
            elif isinstance(motions, np.ndarray):
                print("The npy file contains a standalone NumPy array.")
                print(motions)
                if len(motions.shape) == 3 :
                    print("reshaper called")
                    original_shape = motions.shape
                    motions = motions.reshape((1,) + original_shape)
            else:
                raise("Unknown data type in the npy file.")

            print(filename,'motion reshaped as ', motions.shape)

            render(motions[0], outdir=filedir, device_id=0, name=filename, pred=True)

            # motions = np.load(filedir + filename+'_gt.npy')
            # print('gt', motions.shape, filename)
            # render(motions[0], outdir=filedir, device_id=0, name=filename, pred=False)

            
        print('done')
   