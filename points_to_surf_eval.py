import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from tqdm import tqdm
import sys
#from source.points_to_surf_model import PointsToSurfModel
from source.points_to_surf_model import normalizePointsToSurfModelk4_adddecoder as PointsToSurfModel
from source import test_normalize_data_loader_norot as data_loader
from thop import profile
from torchstat import stat
#from source.points_to_surf_model import multiPointsToSurfModel as PointsToSurfModel
#from source import multi_normal_testdata_loader as data_loader

from source import sdf_nn
from source.base import file_utils
from source.base import utils

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default='/home/mark/下载/pclouds', help='input folder (meshes)')
    parser.add_argument('--outdir', type=str, default='/home/mark/下载/pclouds/unoriented_our',
                        help='output folder (estimated point cloud properties)')
    parser.add_argument('--dataset', nargs='+', type=str, default=['testset_all.txt'], help='shape set file name')
    parser.add_argument('--reconstruction', type=bool, default=False, help='do reconstruction instead of evaluation')
    parser.add_argument('--query_grid_resolution', type=int, default=None,
                        help='resolution of sampled volume used for reconstruction')
    parser.add_argument('--epsilon', type=int, default=-1,
                        help='neighborhood size for reconstruction')
    parser.add_argument('--certainty_threshold', type=float, default=None, help='')
    parser.add_argument('--sigma', type=int, default=None, help='')
    parser.add_argument('--up_sampling_factor', type=int, default=10,
                        help='Neighborhood of points that is queried with the network. '
                             'This enables you to set the trade-off between computation time and tolerance for '
                             'sparsely sampled surfaces.')
    parser.add_argument('--modeldir', type=str, default='/home/mark/Experiment/wsy/our_normals/models', help='model folder')
    parser.add_argument('--models', type=str, default='base_k4_normalize_adddecoder',
                        help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_model.pth', help='model file postfix')
    parser.add_argument('--parampostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    parser.add_argument('--sparse_patches', type=int, default=False,
                        help='evaluate on a sparse set of patches, given by a '
                             '.pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random '
                        'points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000,
                        help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--query_points_per_patch', type=int, default=1,
                        help='number of query points per patch')
    parser.add_argument('--sub_sample_size', type=int, default=1000,
                        help='number of points of the point cloud that are trained with each patch')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=200
                        , help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--outputs', type=str, nargs='+', default=['normals'],
                        help='outputs of the network, a list with elements of:\n'
                             'normals: oriented nomals')
    parser.add_argument('--cache_capacity', type=int, default=100,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean',
                        help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')
    opt = parser.parse_args(args=args)
    if len(opt.dataset) == 1:
        opt.dataset = opt.dataset[0]

    return opt


def get_output_ids(train_opt):
    # get output ids

    oi = {
        'imp': [i for i, o in enumerate(train_opt.outputs) if o == 'imp_surf'],
        'ism': [i for i, o in enumerate(train_opt.outputs) if o == 'imp_surf_magnitude'],
        'iss': [i for i, o in enumerate(train_opt.outputs) if o == 'imp_surf_sign'],
        'pid': [i for i, o in enumerate(train_opt.outputs) if o == 'p_index'],
        'ids': [i for i, o in enumerate(train_opt.outputs) if o == 'patch_pts_ids'],
        'normal':[i for i, o in enumerate(train_opt.outputs) if o == 'normals']
    }
    return oi


def get_output_dimensions(train_opt):
    # get indices in targets and predictions corresponding to each output

    pred_dim = 0
    output_pred_ind = []
    for o in train_opt.outputs:
        if o == 'imp_surf':
            output_pred_ind.append(pred_dim)
            pred_dim += 1
        elif o == 'imp_surf_magnitude':
            output_pred_ind.append(pred_dim)
            pred_dim += 1
        elif o == 'imp_surf_sign':
            output_pred_ind.append(pred_dim)
            pred_dim += 1
        elif o == 'p_index':
            output_pred_ind.append(pred_dim)
        elif o == 'patch_pts_ids':
            output_pred_ind.append(pred_dim)
        elif o =='normals':
            output_pred_ind.append(pred_dim)
            pred_dim+=3
        else:
            raise ValueError('Unknown output: %s' % o)

    return pred_dim, output_pred_ind


def make_dataset(train_opt, eval_opt):
    dataset = data_loader.PointcloudPatchDataset(
        root=eval_opt.indir, shape_list_filename=eval_opt.dataset,
        points_per_patch=train_opt.points_per_patch,
        patch_features=eval_opt.outputs,
        seed=eval_opt.seed,
        center=train_opt.patch_center,
        cache_capacity=eval_opt.cache_capacity,
        pre_processed_patches=True,
        sub_sample_size=train_opt.sub_sample_size,
        reconstruction=eval_opt.reconstruction,
        query_grid_resolution=eval_opt.query_grid_resolution,
        num_workers=int(eval_opt.workers),
        patch_radius=train_opt.patch_radius,
        epsilon=eval_opt.epsilon,  # not necessary for training
        #uniform_subsample=train_opt.uniform_subsample if 'uniform_subsample' in train_opt else 0,
        uniform_subsample=0
    )
    return dataset


def make_datasampler(eval_opt, dataset):
    if eval_opt.sampling == 'full':
        return data_loader.SequentialPointcloudPatchSampler(dataset)
    elif eval_opt.sampling == 'sequential_shapes_random_patches':
        return data_loader.SequentialShapeRandomPointcloudPatchSampler(
            dataset,
            patches_per_shape=eval_opt.patches_per_shape,
            seed=eval_opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % eval_opt.sampling)


def make_dataloader(eval_opt, dataset, datasampler, model_batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=model_batch_size,
        num_workers=int(eval_opt.workers))
    return dataloader


def make_regressor(train_opt, pred_dim, model_filename, device):

    use_query_point = any([f in train_opt.outputs for f in ['normals']])
    p2s_model = PointsToSurfModel(
        net_size_max=train_opt.net_size if 'net_size' in train_opt else 1024,
        num_points=train_opt.points_per_patch,
        output_dim=pred_dim,
        use_point_stn=train_opt.use_point_stn,
        use_feat_stn=train_opt.use_feat_stn,
        sym_op=train_opt.sym_op,
        use_query_point=use_query_point,
        sub_sample_size=train_opt.sub_sample_size,
        do_augmentation=False,
        single_transformer=train_opt.single_transformer,
        shared_transformation=train_opt.shared_transformer,
    )

    p2s_model.cuda(device=device)  # same order as in training
    p2s_model = torch.nn.DataParallel(p2s_model)
    p2s_model.load_state_dict(torch.load(model_filename))
    p2s_model.eval()
    return p2s_model


def post_process(batch_pred, train_opt, output_ids, output_pred_ind, patch_radius, fixed_radius, trans):
    # post-processing of the prediction
    if 'imp_surf' in train_opt.outputs:
        oi_imp = output_ids['imp'][0]
        imp_surf_pred = batch_pred[:, output_pred_ind[oi_imp]:output_pred_ind[oi_imp] + 1]
        imp_surf_pred = sdf_nn.post_process_distance(pred=imp_surf_pred)
        if not fixed_radius:
            imp_surf_pred *= patch_radius.unsqueeze(dim=1)
        batch_pred[:, output_pred_ind[oi_imp]:output_pred_ind[oi_imp] + 1] = \
            imp_surf_pred
    if 'imp_surf_magnitude' in train_opt.outputs:
        oi_ism = output_ids['ism'][0]
        imp_surf_mag_pred = batch_pred[:, output_pred_ind[oi_ism]:output_pred_ind[oi_ism] + 1]
        imp_surf_mag_pred = sdf_nn.post_process_magnitude(pred=imp_surf_mag_pred)
        if not fixed_radius:
            imp_surf_mag_pred *= patch_radius.unsqueeze(dim=1)
        batch_pred[:, output_pred_ind[oi_ism]:output_pred_ind[oi_ism] + 1] = \
            imp_surf_mag_pred
    if 'imp_surf_sign' in train_opt.outputs:
        oi_iss = output_ids['iss'][0]
        imp_surf_sig_pred = batch_pred[:, output_pred_ind[oi_iss]:output_pred_ind[oi_iss] + 1]
        imp_surf_sig_pred = sdf_nn.post_process_sign(pred=imp_surf_sig_pred)
        batch_pred[:, output_pred_ind[oi_iss]:output_pred_ind[oi_iss] + 1] = imp_surf_sig_pred

    if 'normals' in train_opt.outputs:
        oi_n = output_ids['normal'][0]
        n_pred = batch_pred[:, output_pred_ind[oi_n]:output_pred_ind[oi_n] + 3]
        if trans is not None:
            n_pred[:,:] = torch.bmm(n_pred.unsqueeze(1),trans.transpose(2,1)).squeeze(dim=1)
        n_pred_len = torch.max(n_pred.new_tensor([sys.float_info.epsilon * 100]), n_pred.norm(p=2, dim=1, keepdim=True))
        n_pred = n_pred / n_pred_len
        batch_pred[:, output_pred_ind[oi_n]:output_pred_ind[oi_n] + 3] = n_pred


def save_reconstruction_data(imp_surf_dist_ms, dataset, model_out_dir, shape_ind):

    from source import sdf

    shape = dataset.shape_cache.get(shape_ind)

    imp_surf_dist_ms_nan = np.isnan(imp_surf_dist_ms)
    # the predicted distance would be greater than 1 -> not possible with tanh
    imp_surf_dist_ms[imp_surf_dist_ms_nan] = 1.0

    # save query points
    os.makedirs(os.path.join(model_out_dir, 'query_pts_ms'), exist_ok=True)
    np.save(os.path.join(model_out_dir, 'query_pts_ms', dataset.shape_names[shape_ind] + '.xyz.npy'),
            shape.imp_surf_query_point_ms)

    # save query distance in model space
    os.makedirs(os.path.join(model_out_dir, 'dist_ms'), exist_ok=True)
    np.save(os.path.join(model_out_dir, 'dist_ms', dataset.shape_names[shape_ind] + '.xyz.npy'), imp_surf_dist_ms)

    # debug query points with color for distance
    os.makedirs(os.path.join(model_out_dir, 'query_pts_ms_vis'), exist_ok=True)
    sdf.visualize_query_points(
        query_pts_ms=shape.imp_surf_query_point_ms, query_dist_ms=imp_surf_dist_ms,
        file_out_off=os.path.join(model_out_dir, 'query_pts_ms_vis', dataset.shape_names[shape_ind] + '.ply'))


def save_evaluation(datasampler, dataset, eval_opt, model_out_dir, output_ids, output_pred_ind, shape_ind,
                    shape_patch_values, train_opt):
    # save shape properties to disk
    prop_saved = [False] * len(train_opt.outputs)

    def visualize_result(pts_query_ms, dist_query_ms):
        out_vis_file = os.path.join(model_out_dir, 'vis', dataset.shape_names[shape_ind] + '.ply')
        file_utils.make_dir_for_file(out_vis_file)
        from source import sdf
        sdf.visualize_query_points(pts_query_ms, dist_query_ms, out_vis_file)

    # save implicit surface
    if len(output_ids['imp']) > 1:
        raise ValueError('Duplicate implicit surface output.')
    elif len(output_ids['imp']) == 1:
        oid_imp = output_pred_ind[output_ids['imp'][0]]
        imp_surf_shape_ms = shape_patch_values[:, oid_imp:oid_imp + 1]
        imp_surf_shape_ms = imp_surf_shape_ms.squeeze()

        imp_surf_np_ms = imp_surf_shape_ms.cpu().numpy()

        if eval_opt.reconstruction:
            save_reconstruction_data(imp_surf_np_ms, dataset, model_out_dir, shape_ind)

        os.makedirs(os.path.join(model_out_dir, 'eval'), exist_ok=True)
        np.save(os.path.join(model_out_dir, 'eval', dataset.shape_names[shape_ind] + '.xyz.npy'), imp_surf_np_ms)
        np.savetxt(os.path.join(model_out_dir, 'eval', dataset.shape_names[shape_ind] + '.xyz.txt'), imp_surf_np_ms)
        visualize_result(pts_query_ms=dataset.shape_cache.get(shape_ind).imp_surf_query_point_ms,
                         dist_query_ms=imp_surf_np_ms)

        prop_saved[output_ids['imp'][0]] = True

    if len(output_ids['ism']) > 1:
        raise ValueError('Duplicate implicit surface magnitude output.')
    elif len(output_ids['ism']) == 1:
        if len(output_ids['iss']) != 1:
            raise ValueError('Implicit surface magnitude output without sign!')

        oid_ism = output_pred_ind[output_ids['ism'][0]]
        imp_surf_mag_shape_ms = shape_patch_values[:, oid_ism:oid_ism + 1]
        imp_surf_mag_shape_ms = imp_surf_mag_shape_ms.squeeze()

        oid_iss = output_pred_ind[output_ids['iss'][0]]
        imp_surf_sig_shape = shape_patch_values[:, oid_iss:oid_iss + 1]
        imp_surf_sig_shape = imp_surf_sig_shape.squeeze()

        imp_surf_shape_ms = imp_surf_mag_shape_ms * imp_surf_sig_shape

        imp_surf_np_ms = imp_surf_shape_ms.cpu().numpy()

        os.makedirs(os.path.join(model_out_dir, 'eval'), exist_ok=True)
        np.save(os.path.join(model_out_dir, 'eval', dataset.shape_names[shape_ind] + '.xyz.npy'), imp_surf_np_ms)
        np.savetxt(os.path.join(model_out_dir, 'eval', dataset.shape_names[shape_ind] + '.xyz.txt'), imp_surf_np_ms)
        visualize_result(pts_query_ms=dataset.shape_cache.get(shape_ind).imp_surf_query_point_ms,
                         dist_query_ms=imp_surf_np_ms)

        if eval_opt.reconstruction:
            save_reconstruction_data(imp_surf_np_ms, dataset, model_out_dir, shape_ind)

        prop_saved[output_ids['ism'][0]] = True
        prop_saved[output_ids['iss'][0]] = True

    if len(output_ids['normal']) >1:
        raise ValueError('Duplicate normal output.')
    elif len(output_ids['normal']) ==1:
        oi=output_ids['normal'][0]
        normal_prop = shape_patch_values[:, output_pred_ind[oi]:output_pred_ind[oi] + 3]
        #normal_prop=shape_patch_values[:,1:4]
        #label_prop=shape_patch_values[:,0:1]
        np.savetxt(os.path.join(model_out_dir, dataset.shape_names[shape_ind] + '.normals'), normal_prop.cpu().numpy())
        #np.savetxt(os.path.join(model_out_dir, dataset.shape_names[shape_ind] + '.labels'), label_prop.cpu().numpy())
        prop_saved[oi] = True
    ##prop_saved[output_ids['ids'][0]] = True
    ##prop_saved[output_ids['pid'][0]] = True
    if not all(prop_saved):
        raise ValueError('Not all shape properties were saved, some of them seem to be unsupported.')
    ## save point indices
    if eval_opt.sampling != 'full':
        np.savetxt(os.path.join(model_out_dir, dataset.shape_names[shape_ind] + '.idx'),
                   datasampler.shape_patch_inds[shape_ind], fmt='%d')

def compute_loss(pred, batch_data, outputs, normal_loss):
    n_loss=0
    #wsy:add normal loss(same as pcpnet)
    if 'normals' in outputs:
        o_pred=pred
        o_target= batch_data['normals']
        if normal_loss == 'ms_euclidean':
            n_loss += torch.sqrt((o_pred - o_target).pow(2).sum(1).mean())
        elif normal_loss == 'ms_oneminuscos':
            n_loss += (1 - utils.cos_angle(o_pred, o_target)).pow(2).mean()
        else:
            raise ValueError('Unsupported loss type: %s' % (normal_loss))

    return n_loss

def compute_rms_loss(pred, batch_data):
    loss=0
    abs_loss=0
    right=0.0
    o_target= batch_data['normals']

    #angle_dif = torch.acos(utils.cos_angle(pred, o_target)) / 3.14 * 180
    #abs_angle_diff=torch.acos(abs(utils.cos_angle(pred, o_target))) / 3.14 * 180
    angle_dif = torch.acos(utils.cos_angle(pred, o_target))
    abs_angle_diff=torch.acos(abs(utils.cos_angle(pred, o_target)))

    loss += torch.sqrt(torch.mean(angle_dif.pow(2)))
    abs_loss+=torch.sqrt(torch.mean(abs_angle_diff.pow(2)))

    angle = angle_dif/3.14*180
    angle = angle<90
    right += float(sum(angle))
    acc=float(right/200)

    return loss,acc,abs_loss,angle

def points_to_surf_eval(eval_opt):

    models = eval_opt.models.split()

    if eval_opt.seed < 0:
        eval_opt.seed = random.randint(1, 10000)

    device = torch.device("cpu" if eval_opt.gpu_idx < 0 else "cuda:%d" % eval_opt.gpu_idx)

    for model_name in models:

        print("Random Seed: %d" % eval_opt.seed)
        random.seed(eval_opt.seed)
        torch.manual_seed(eval_opt.seed)

        model_filename = os.path.join(eval_opt.modeldir, model_name+eval_opt.modelpostfix)
        param_filename = os.path.join(eval_opt.modeldir, model_name+eval_opt.parampostfix)

        # load model and training parameters
        train_opt = torch.load(param_filename)
        if not hasattr(train_opt, 'single_transformer'):
            train_opt.single_transformer = 0
        if not hasattr(train_opt, 'shared_transformer'):
            train_opt.shared_transformation = False

        output_ids = get_output_ids(train_opt)
        
        if eval_opt.batchSize == 0:
            model_batch_size = train_opt.batchSize
        else:
            model_batch_size = eval_opt.batchSize

        pred_dim, output_pred_ind = get_output_dimensions(train_opt)
        dataset = make_dataset(train_opt=train_opt, eval_opt=eval_opt)
        datasampler = make_datasampler(eval_opt=eval_opt, dataset=dataset)
        dataloader = make_dataloader(eval_opt=eval_opt, dataset=dataset, datasampler=datasampler,
                                     model_batch_size=model_batch_size)
        p2s_model = make_regressor(train_opt=train_opt, pred_dim=pred_dim, model_filename=model_filename, device=device)
        shape_ind = 0
        shape_patch_offset = 0
        if eval_opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif eval_opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(eval_opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s'.format(eval_opt.sampling))

        shape_patch_values = torch.zeros(shape_patch_count, pred_dim,
                                         dtype=torch.float32, device=device)

        # append model name to output dipatches_per_shaperectory and create directory if necessary
        if eval_opt.reconstruction:
            model_out_dir = os.path.join(eval_opt.outdir, 'rec')
        else:
            model_out_dir = os.path.join(eval_opt.outdir, 'eval')
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)

        print(f'evaluating {len(dataset)} patches')
        for batch_data in tqdm(dataloader):

            # batch data to GPU
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].cuda(non_blocking=True)

            fixed_radius = train_opt.patch_radius > 0.0
            patch_radius = train_opt.patch_radius

            if not fixed_radius:
                patch_radius = batch_data['patch_radius_ms']

            with torch.no_grad():
                _,batch_pred,trans,_ = p2s_model(batch_data)
                #batch_pred=sdf_nn.post_process_sign(batch_pred)
                #batch_pred=local_normal*batch_pred
            #flops,params=profile(p2s_model,inputs=(batch_data))
            #print('flops:',str(flops/1e9)+'{}'.format("G"))
            #print('params:',str(params/1e6)+'{}'.format("M"))
            #stat(p2s_model)
            post_process(batch_pred, train_opt, output_ids, output_pred_ind, patch_radius, fixed_radius,trans)


            batch_offset = 0
            #batch_pred=torch.unsqueeze(angle_right_label,dim=1)
            #batch_pred=torch.cat((torch.unsqueeze(angle_right_label,dim=1).float(),batch_pred,),dim=1)
            while batch_offset < batch_pred.size(0):

                shape_patches_remaining = shape_patch_count-shape_patch_offset
                batch_patches_remaining = batch_pred.size(0)-batch_offset
                samples_remaining = min(shape_patches_remaining, batch_patches_remaining)

                # append estimated patch properties batch to properties for the current shape
                patch_properties = batch_pred[batch_offset:batch_offset+samples_remaining]

                shape_patch_values[shape_patch_offset:shape_patch_offset+samples_remaining] = patch_properties
            
                batch_offset = batch_offset + samples_remaining
                shape_patch_offset = shape_patch_offset + samples_remaining

                if shape_patches_remaining <= batch_patches_remaining:
                    save_evaluation(datasampler, dataset, eval_opt, model_out_dir, output_ids, output_pred_ind,
                                    shape_ind, shape_patch_values, train_opt)

                    # start new shape
                    if shape_ind + 1 < len(dataset.shape_names):
                        shape_patch_offset = 0
                        shape_ind = shape_ind + 1
                        if eval_opt.sampling == 'full':
                            shape_patch_count = dataset.shape_patch_count[shape_ind]
                        elif eval_opt.sampling == 'sequential_shapes_random_patches':
                            shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                        else:
                            raise ValueError('Unknown sampling strategy: %s' % eval_opt.sampling)
                        shape_patch_values = torch.zeros(shape_patch_count, pred_dim,
                                                         dtype=torch.float32, device=device)


if __name__ == '__main__':
    eval_opt = parse_arguments()
    points_to_surf_eval(eval_opt)

