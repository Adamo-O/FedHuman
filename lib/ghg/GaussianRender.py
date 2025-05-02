import torch
from gaussian_renderer import render, inputRender
import pdb

def pts2render(data, bg_color, phase='train'):
    
    bs = data['pos'].shape[0]

    render_novel_list = []
    render_depth_list = []
    render_alpha_list = []

    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []

        for shell in ['in_shell', 'out_shell_1', 'out_shell_2', 'out_shell_3', 'out_shell_4']:

            
            valid_i = data[shell]['pts_valid'][i, :]
            xyz_i = data[shell]['xyz'][i, :, :]
            
            
            rgb_i = data[shell]['rgb_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            
            rot_i = data[shell]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[shell]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1,3)

            if (shell == 'in_shell') and (phase == 'test'):
                scale_i = torch.clamp_min(scale_i,0.0013)

            opacity_i = data[shell]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            
            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            

            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        
        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        
        
        
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
      
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        multi_view_renders = []
        multi_depth_renders = []
        multi_alpha_renders = []

        if phase == 'test':
            novel_view_name_list = ['novel_view']
        elif phase == 'train':
            novel_view_name_list = ['novel_view_0', 'novel_view_1', 'novel_view_2']

        
        # exit("Exiting in pts2render")
        for novel_view in novel_view_name_list:
            
            render_novel_view_i, render_novel_depth_i, render_novel_alpha_i \
                = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i,
                         opacity_i, bg_color=bg_color,
                         novel_view_name=novel_view)

            multi_view_renders.append(render_novel_view_i)
            multi_depth_renders.append(render_novel_depth_i)
            multi_alpha_renders.append(render_novel_alpha_i)

        # for multi-view supervision
        multi_view_renders_tensor = torch.stack(multi_view_renders, 0)
        multi_depth_renders_tensor = torch.stack(multi_depth_renders, 0)
        multi_alpha_renders_tensor = torch.stack(multi_alpha_renders, 0)

        render_novel_list.append(multi_view_renders_tensor.unsqueeze(0))
        render_depth_list.append(multi_depth_renders_tensor.unsqueeze(0))
        render_alpha_list.append(multi_alpha_renders_tensor.unsqueeze(0))


    if phase == 'test':

        predictions = torch.concat(render_novel_list, dim=0)
        data['novel_view']['img_pred'] = predictions[:, 0]

        depth_predictions = torch.concat(render_depth_list, dim=0)
        data['novel_view']['depth_pred'] = depth_predictions[:, 0]

        alpha_predictions = torch.concat(render_alpha_list, dim=0)
        data['novel_view']['alpha_pred'] = alpha_predictions[:, 0]

    elif phase == 'train':

        predictions = torch.concat(render_novel_list,dim=0)
        data['novel_view_0']['img_pred'] = predictions[:, 0]
        data['novel_view_1']['img_pred'] = predictions[:, 1]
        data['novel_view_2']['img_pred'] = predictions[:, 2]

        depth_predictions = torch.concat(render_depth_list, dim=0)
        data['novel_view_0']['depth_pred'] = depth_predictions[:, 0]
        data['novel_view_1']['depth_pred'] = depth_predictions[:, 1]
        data['novel_view_2']['depth_pred'] = depth_predictions[:, 2]

        alpha_predictions = torch.concat(render_alpha_list, dim=0)
        data['novel_view_0']['alpha_pred'] = alpha_predictions[:, 0]
        data['novel_view_1']['alpha_pred'] = alpha_predictions[:, 1]
        data['novel_view_2']['alpha_pred'] = alpha_predictions[:, 2]

    return data


def input2Render(data, bg_color, phase='train'):
    
    bs = data['pos'].shape[0]

    render_input_list = []
    render_input_depth_list = []
    render_input_alpha_list = []

    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []

        for shell in ['in_shell', 'out_shell_1', 'out_shell_2', 'out_shell_3', 'out_shell_4']:

            
            valid_i = data[shell]['pts_valid'][i, :]
            xyz_i = data[shell]['xyz'][i, :, :]
            
            
            rgb_i = data[shell]['rgb_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            
            rot_i = data[shell]['input_rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[shell]['input_scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1,3)

            if (shell == 'in_shell') and (phase == 'test'):
                scale_i = torch.clamp_min(scale_i,0.0013)

            opacity_i = data[shell]['input_opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            
            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            

            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

       
        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
      
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        multi_view_renders = []
        multi_depth_renders = []
        multi_alpha_renders = []

        
            
        
        input_view_name_list = ['input_view_0', 'input_view_1', 'input_view_2']

        
        # exit("Exiting in pts2render")
        for input_view in input_view_name_list:
            
            render_input_view_i, render_input_depth_i, render_input_alpha_i \
                = inputRender(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i,
                         opacity_i, bg_color=bg_color,
                         input_view_name=input_view)

            multi_view_renders.append(render_input_view_i)
            multi_depth_renders.append(render_input_depth_i)
            multi_alpha_renders.append(render_input_alpha_i)

        # for multi-view supervision
        multi_view_renders_tensor = torch.stack(multi_view_renders, 0)
        multi_depth_renders_tensor = torch.stack(multi_depth_renders, 0)
        multi_alpha_renders_tensor = torch.stack(multi_alpha_renders, 0)

        render_input_list.append(multi_view_renders_tensor.unsqueeze(0))
        render_input_depth_list.append(multi_depth_renders_tensor.unsqueeze(0))
        render_input_alpha_list.append(multi_alpha_renders_tensor.unsqueeze(0))


    


    

    predictions = torch.concat(render_input_list,dim=0)
    data['input_view']['input_view_0']['img_pred'] = predictions[:, 0]
    data['input_view']['input_view_1']['img_pred'] = predictions[:, 1]
    data['input_view']['input_view_2']['img_pred'] = predictions[:, 2]

    depth_predictions = torch.concat(render_input_depth_list, dim=0)
    data['input_view']['input_view_0']['depth_pred'] = depth_predictions[:, 0]
    data['input_view']['input_view_1']['depth_pred'] = depth_predictions[:, 1]
    data['input_view']['input_view_2']['depth_pred'] = depth_predictions[:, 2]

    alpha_predictions = torch.concat(render_input_alpha_list, dim=0)
    data['input_view']['input_view_0']['alpha_pred'] = alpha_predictions[:, 0]
    data['input_view']['input_view_1']['alpha_pred'] = alpha_predictions[:, 1]
    data['input_view']['input_view_2']['alpha_pred'] = alpha_predictions[:, 2]
        
    

    return data


def multiRender(data, bg_color, phase='train'):
    """
    Combined function that renders both novel views and input views in one go.
    This merges the functionality of pts2render and inputRender.
    
    Args:
        data: Dictionary containing all the necessary data
        bg_color: Background color for rendering
        phase: 'train' or 'test' phase
        
    Returns:
        data: Updated data dictionary with rendered images
    """
    bs = data['pos'].shape[0]
    
    # Lists for novel view rendering results
    render_novel_list = []
    render_depth_list = []
    render_alpha_list = []
    
    # Lists for input view rendering results
    render_input_list = []
    render_input_depth_list = []
    render_input_alpha_list = []
    
    for i in range(bs):
        # For novel view rendering
        novel_xyz_i_valid = []
        novel_rgb_i_valid = []
        novel_rot_i_valid = []
        novel_scale_i_valid = []
        novel_opacity_i_valid = []
        
        # For input view rendering
        input_xyz_i_valid = []
        input_rgb_i_valid = []
        input_rot_i_valid = []
        input_scale_i_valid = []
        input_opacity_i_valid = []
        
        for shell in ['in_shell', 'out_shell_1', 'out_shell_2', 'out_shell_3', 'out_shell_4']:
            # Common data for both novel and input views
            valid_i = data[shell]['pts_valid'][i, :]
            xyz_i = data[shell]['xyz'][i, :, :]
            rgb_i = data[shell]['rgb_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            
            # Novel view specific data
            novel_rot_i = data[shell]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            novel_scale_i = data[shell]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            novel_opacity_i = data[shell]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            
            # Input view specific data
            input_rot_i = data[shell]['input_rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            input_scale_i = data[shell]['input_scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            input_opacity_i = data[shell]['input_opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            
            # Apply min scale clamp for test phase if needed
            if (shell == 'in_shell') and (phase == 'test'):
                novel_scale_i = torch.clamp_min(novel_scale_i, 0.0013)
                input_scale_i = torch.clamp_min(input_scale_i, 0.0013)
            
            # Filter by valid points and append to lists
            xyz_valid = xyz_i[valid_i].view(-1, 3)
            rgb_valid = rgb_i[valid_i].view(-1, 3)
            
            # Novel view lists
            novel_xyz_i_valid.append(xyz_valid)
            novel_rgb_i_valid.append(rgb_valid)
            novel_rot_i_valid.append(novel_rot_i[valid_i].view(-1, 4))
            novel_scale_i_valid.append(novel_scale_i[valid_i].view(-1, 3))
            novel_opacity_i_valid.append(novel_opacity_i[valid_i].view(-1, 1))
            
            # Input view lists
            input_xyz_i_valid.append(xyz_valid)
            input_rgb_i_valid.append(rgb_valid)
            input_rot_i_valid.append(input_rot_i[valid_i].view(-1, 4))
            input_scale_i_valid.append(input_scale_i[valid_i].view(-1, 3))
            input_opacity_i_valid.append(input_opacity_i[valid_i].view(-1, 1))
        
        # Concatenate novel view tensors
        novel_pts_xyz_i = torch.concat(novel_xyz_i_valid, dim=0)
        novel_pts_rgb_i = torch.concat(novel_rgb_i_valid, dim=0)
        novel_pts_rgb_i = novel_pts_rgb_i * 0.5 + 0.5  # Scale RGB values
        novel_rot_i = torch.concat(novel_rot_i_valid, dim=0)
        novel_scale_i = torch.concat(novel_scale_i_valid, dim=0)
        novel_opacity_i = torch.concat(novel_opacity_i_valid, dim=0)
        
        # Concatenate input view tensors
        input_pts_xyz_i = torch.concat(input_xyz_i_valid, dim=0)
        input_pts_rgb_i = torch.concat(input_rgb_i_valid, dim=0)
        input_pts_rgb_i = input_pts_rgb_i * 0.5 + 0.5  # Scale RGB values
        input_rot_i = torch.concat(input_rot_i_valid, dim=0)
        input_scale_i = torch.concat(input_scale_i_valid, dim=0)
        input_opacity_i = torch.concat(input_opacity_i_valid, dim=0)
        
        # Render novel views
        novel_multi_view_renders = []
        novel_multi_depth_renders = []
        novel_multi_alpha_renders = []
        
        if phase == 'test':
            novel_view_name_list = ['novel_view']
        elif phase == 'train':
            novel_view_name_list = ['novel_view_0', 'novel_view_1', 'novel_view_2']
            
        for novel_view in novel_view_name_list:
            render_novel_view_i, render_novel_depth_i, render_novel_alpha_i = render(
                data, i, novel_pts_xyz_i, novel_pts_rgb_i, novel_rot_i, novel_scale_i,
                novel_opacity_i, bg_color=bg_color, novel_view_name=novel_view
            )
            
            novel_multi_view_renders.append(render_novel_view_i)
            novel_multi_depth_renders.append(render_novel_depth_i)
            novel_multi_alpha_renders.append(render_novel_alpha_i)
            
        # Stack and store novel view results
        novel_multi_view_renders_tensor = torch.stack(novel_multi_view_renders, 0)
        novel_multi_depth_renders_tensor = torch.stack(novel_multi_depth_renders, 0)
        novel_multi_alpha_renders_tensor = torch.stack(novel_multi_alpha_renders, 0)
        
        render_novel_list.append(novel_multi_view_renders_tensor.unsqueeze(0))
        render_depth_list.append(novel_multi_depth_renders_tensor.unsqueeze(0))
        render_alpha_list.append(novel_multi_alpha_renders_tensor.unsqueeze(0))
        
        # Render input views
        input_multi_view_renders = []
        input_multi_depth_renders = []
        input_multi_alpha_renders = []
        
        input_view_name_list = ['input_view_0', 'input_view_1', 'input_view_2']
        
        for input_view in input_view_name_list:
            render_input_view_i, render_input_depth_i, render_input_alpha_i = render(
                data, i, input_pts_xyz_i, input_pts_rgb_i, input_rot_i, input_scale_i,
                input_opacity_i, bg_color=bg_color, novel_view_name=input_view
            )
            
            input_multi_view_renders.append(render_input_view_i)
            input_multi_depth_renders.append(render_input_depth_i)
            input_multi_alpha_renders.append(render_input_alpha_i)
            
        # Stack and store input view results
        input_multi_view_renders_tensor = torch.stack(input_multi_view_renders, 0)
        input_multi_depth_renders_tensor = torch.stack(input_multi_depth_renders, 0)
        input_multi_alpha_renders_tensor = torch.stack(input_multi_alpha_renders, 0)
        
        render_input_list.append(input_multi_view_renders_tensor.unsqueeze(0))
        render_input_depth_list.append(input_multi_depth_renders_tensor.unsqueeze(0))
        render_input_alpha_list.append(input_multi_alpha_renders_tensor.unsqueeze(0))
    
    # Update data dictionary with novel view results
    if phase == 'test':
        predictions = torch.concat(render_novel_list, dim=0)
        data['novel_view']['img_pred'] = predictions[:, 0]
        
        depth_predictions = torch.concat(render_depth_list, dim=0)
        data['novel_view']['depth_pred'] = depth_predictions[:, 0]
        
        alpha_predictions = torch.concat(render_alpha_list, dim=0)
        data['novel_view']['alpha_pred'] = alpha_predictions[:, 0]
    elif phase == 'train':
        predictions = torch.concat(render_novel_list, dim=0)
        data['novel_view_0']['img_pred'] = predictions[:, 0]
        data['novel_view_1']['img_pred'] = predictions[:, 1]
        data['novel_view_2']['img_pred'] = predictions[:, 2]
        
        depth_predictions = torch.concat(render_depth_list, dim=0)
        data['novel_view_0']['depth_pred'] = depth_predictions[:, 0]
        data['novel_view_1']['depth_pred'] = depth_predictions[:, 1]
        data['novel_view_2']['depth_pred'] = depth_predictions[:, 2]
        
        alpha_predictions = torch.concat(render_alpha_list, dim=0)
        data['novel_view_0']['alpha_pred'] = alpha_predictions[:, 0]
        data['novel_view_1']['alpha_pred'] = alpha_predictions[:, 1]
        data['novel_view_2']['alpha_pred'] = alpha_predictions[:, 2]
    
    # Update data dictionary with input view results
    input_predictions = torch.concat(render_input_list, dim=0)
    data['input_view']['input_view_0']['img_pred'] = input_predictions[:, 0]
    data['input_view']['input_view_1']['img_pred'] = input_predictions[:, 1]
    data['input_view']['input_view_2']['img_pred'] = input_predictions[:, 2]
    
    input_depth_predictions = torch.concat(render_input_depth_list, dim=0)
    data['input_view']['input_view_0']['depth_pred'] = input_depth_predictions[:, 0]
    data['input_view']['input_view_1']['depth_pred'] = input_depth_predictions[:, 1]
    data['input_view']['input_view_2']['depth_pred'] = input_depth_predictions[:, 2]
    
    input_alpha_predictions = torch.concat(render_input_alpha_list, dim=0)
    data['input_view']['input_view_0']['alpha_pred'] = input_alpha_predictions[:, 0]
    data['input_view']['input_view_1']['alpha_pred'] = input_alpha_predictions[:, 1]
    data['input_view']['input_view_2']['alpha_pred'] = input_alpha_predictions[:, 2]
    
    return data


def features2render(data, bg_color, phase='train',appearence_map=False):
    
    bs = data['pos'].shape[0]
    render_novel_list = []

    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []

        for shell in ['in_shell', 'out_shell_1', 'out_shell_2', 'out_shell_3', 'out_shell_4']:
            # Deep copy everything except features
            valid_i = data[shell]['pts_valid'][i, :].detach()
            xyz_i = data[shell]['xyz'][i, :, :].detach()
            
            # Keep gradients only for features
            if appearence_map:
                rgb_i = data[shell]['appearence_map'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            else:
                rgb_i = data[shell]['features'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            
            rot_i = data[shell]['rot_maps'][i, :, :, :].detach().permute(1, 2, 0).view(-1, 4)
            scale_i = data[shell]['scale_maps'][i, :, :, :].detach().permute(1, 2, 0).view(-1,3)

            if (shell == 'in_shell') and (phase == 'test'):
                scale_i = torch.clamp_min(scale_i,0.0013)

            opacity_i = data[shell]['opacity_maps'][i, :, :, :].detach().permute(1, 2, 0).view(-1, 1)
            
            xyz_i_valid.append(xyz_i.view(-1, 3))
            rgb_i_valid.append(rgb_i.view(-1, 3))
            rot_i_valid.append(rot_i.view(-1, 4))
            scale_i_valid.append(scale_i.view(-1, 3))
            opacity_i_valid.append(opacity_i.view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
      
        # pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        input_view_name_list = ['input_view_0', 'input_view_1', 'input_view_2']
       
        new_opacity = torch.ones(opacity_i.shape, device=opacity_i.device, dtype=opacity_i.dtype) * 0.999
        scale_i = torch.ones(scale_i.shape, device=scale_i.device, dtype=scale_i.dtype) * 0.001
        rot_i = torch.zeros_like(rot_i)
        rot_i[:, 0] = 1.0

        features = []
        for input_view in input_view_name_list:
            
            
            render_novel_view_i, _, _ = inputRender(data, i, pts_xyz_i.float(), pts_rgb_i.float(), rot_i.float(), scale_i.float(),
                         new_opacity.float(), bg_color=bg_color,
                         input_view_name=input_view)
            
            features.append(render_novel_view_i)
        
        features = torch.stack(features)

    return features