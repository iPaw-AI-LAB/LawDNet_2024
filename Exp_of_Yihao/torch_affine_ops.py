import torch
import numpy as np
import torch.nn.functional as F

# def transform_torch(center, output_size, scale, rotation):

#     device = center.device
#     Batch_size = center.shape[0]

#     rot = rotation * torch.pi / 180.0
#     #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
#     scale_matrix = torch.eye(3,device=device).repeat(Batch_size,1,1)*scale.view(Batch_size,1,1)
#     scale_matrix[:,-1,-1] =1
#     center_scale = center.view(Batch_size,2)*scale.view(Batch_size,1)

#     scale_matrix[:,:2,-1] = - center_scale
#     #transform_matrix = scale_matrix - torch.tensor([[0,0,center_scale[0]],[0,0,center_scale[1]],[0,0,0]]).to(device).repeat(Batch_size,1,1)

#     #transform_matrix = scale_matrix - transform_matrix
#     rotation_matrix = torch.cat([torch.cos(rot).view(Batch_size,1), -torch.sin(rot).view(Batch_size,1), torch.zeros(Batch_size,1).to(device), torch.sin(rot).view(Batch_size,1), torch.cos(rot).view(Batch_size,1), torch.zeros(Batch_size,1).to(device), torch.zeros(Batch_size,3).to(device)], dim=-1).view(Batch_size,3,3)
#     rotation_matrix[:,-1,-1] = 1

#     #rotation_matrix = torch.tensor([[torch.cos(rot), -torch.sin(rot), 0], [torch.sin(rot), torch.cos(rot), 0], [0, 0, 1]]).to(device)
#     transform_matrix = torch.matmul(rotation_matrix,scale_matrix)
#     recenter_matrix = torch.tensor([[1,0,output_size[0]/2],[0,1,output_size[1]/2],[0,0,1]]).to(device).repeat(Batch_size,1,1)
#     transform_matrix = torch.matmul(recenter_matrix,transform_matrix)

#     return transform_matrix

def standard_grid(size,batch_size=1,device='cuda'):
    """
    equivalent to 
    grid_trans = torch.eye(4).unsqueeze(0)
    F.affine_grid(grid_trans[:,:3,:], torch.Size((1, 3, D,H,W)))
    but more efficient and flexible

    size: (H,W) or (D,H,W)

    return: (B,H,W,2) or (B,D,H,W,3)

    """

    dim = len(size)

    axis = []
    for i in size:
        tmp = torch.linspace(-1+1/i, 1-1/i, i, device=device)
        
        axis.append(tmp)
    
    grid = torch.stack(torch.meshgrid(axis), dim=-1)

    grid = torch.flip(grid, dims=[-1]).contiguous()

    batch_grid = grid.unsqueeze(0).repeat((batch_size,)+(1,)*(dim+1))

    return batch_grid

def transform_torch(center, output_size, scale, rotation):

    device = center.device
    Batch_size = center.shape[0]

    rot = rotation * torch.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    scale_matrix = torch.eye(3,device=device).view(1,3,3).repeat(Batch_size,1,1)*scale.view(Batch_size,1,1)

    scale_matrix[:,-1,-1] = 1

    center_scale = center.view(Batch_size,2)*scale.view(Batch_size,1)

    scale_matrix[:,:2,-1] = - center_scale
    #transform_matrix = scale_matrix - torch.tensor([[0,0,center_scale[0]],[0,0,center_scale[1]],[0,0,0]]).to(device).repeat(Batch_size,1,1)

    #transform_matrix = scale_matrix - transform_matrix
    rotation_matrix = torch.cat([torch.cos(rot).view(Batch_size,1), -torch.sin(rot).view(Batch_size,1), torch.zeros(Batch_size,1).to(device), torch.sin(rot).view(Batch_size,1), torch.cos(rot).view(Batch_size,1), torch.zeros(Batch_size,1).to(device), torch.zeros(Batch_size,3).to(device)], dim=-1).view(Batch_size,3,3)
    rotation_matrix[:,-1,-1] = 1

    #rotation_matrix = torch.tensor([[torch.cos(rot), -torch.sin(rot), 0], [torch.sin(rot), torch.cos(rot), 0], [0, 0, 1]]).to(device)
    transform_matrix = torch.matmul(rotation_matrix,scale_matrix)
    recenter_matrix = torch.tensor([[1,0,output_size[0]/2],[0,1,output_size[1]/2],[0,0,1]]).to(device).repeat(Batch_size,1,1)
    transform_matrix = torch.matmul(recenter_matrix,transform_matrix)

    return transform_matrix



def warp_img_torch(img, transform_matrix, output_size):
    device = img.device
    B, C, H, W = img.shape
    T = torch.Tensor([[2 / (W-1), 0, -1],
              [0, 2 / (H-1), -1],
              [0, 0, 1]]).to(device).repeat(B,1,1)
    
    T2 = torch.Tensor([[2 / (output_size[0]-1), 0, -1],[0, 2 / (output_size[1]-1), -1],[0, 0, 1]]).to(device).repeat(B,1,1)
    M_torch = torch.matmul(T2,torch.matmul(transform_matrix,torch.linalg.inv(T)))
    grid_trans = torch.linalg.inv(M_torch)[:,0:2,:]

    grid = F.affine_grid(grid_trans, torch.Size((B, C, output_size[0], output_size[1])))
    img = F.grid_sample(img, grid)
    return img

def SimilarityTransform_torch_2D(src, dst):
    """
    Solve the least-squares problem to search for the best similarity transformation 2D-2D
    src: B x N x 2

    """

    
    device = src.device
    Batch_size = src.shape[0]

    
    
    assert src.shape[-2] == dst.shape[-2], "number of corresponding points must match"
    assert src.shape[-1] ==2 and dst.shape[-1] ==2, "only 2D points are supported"

    # do demean and standardize for src and dst to enhance the numerical stability
    src_mean = src.mean(-2,keepdim=True)

    src_demean = src - src_mean
    src_std = src_demean.std(-2,keepdim=True).mean(-1,keepdim=True)
    src_demean = src_demean / src_std

    dst_mean = dst.mean(-2,keepdim=True)
    dst_demean = dst - dst_mean
    dst_std = dst_demean.std(-2,keepdim=True).mean(-1,keepdim=True)
    dst_demean = dst_demean / dst_std

    # solve the least square problem

    Q_k1 = src_demean.view(Batch_size,50)
    Q_k2 = (torch.flip(src_demean, dims=[-1])*(torch.tensor([1.,-1.]).to(device))).view(Batch_size,50)
    Q_k3 = torch.tensor([1.,0.]).to(device).repeat(Batch_size,25).view(Batch_size,50)
    Q_k4 = torch.tensor([0.,1.]).to(device).repeat(Batch_size,25).view(Batch_size,50)
    Q = torch.stack((Q_k1,Q_k2,Q_k3,Q_k4),dim=-1)

    #P = dst_demean.view(Batch_size,50,1)
  
    P = dst_demean.view(Batch_size,50,1)
    S = torch.matmul(Q.transpose(-1,-2),P)
    QTQ = torch.matmul(Q.transpose(-1,-2),Q)
    M = torch.matmul(torch.inverse(QTQ+1e-8*torch.eye(4).to(device)),S)
    Similarity_Matrix_1k = torch.cat((M[:,0,:],M[:,1,:],M[:,2,:]),dim=-1)
    Similarity_Matrix_2k = torch.cat((-M[:,1,:],M[:,0,:],M[:,3,:]),dim=-1)
    Similarity_Matrix = torch.stack((Similarity_Matrix_1k,Similarity_Matrix_2k),dim=-2).view(Batch_size,2,3)
    
    # recover the scale and translation


    
    trans_matrix_1 = torch.cat([torch.eye(2,device=device).unsqueeze(0).repeat(Batch_size,1,1),-src_mean.transpose(-1,-2)],dim=-1)/src_std
    
    trans_matrix_2 = torch.cat([torch.eye(2,device=device).unsqueeze(0).repeat(Batch_size,1,1),-dst_mean.transpose(-1,-2)],dim=-1)/dst_std

    trans_matrix_3 = torch.tensor([0.,0.,1.],device=device).unsqueeze(0).repeat(Batch_size,1,1)
    
    Similarity_Matrix_3d = torch.cat([Similarity_Matrix,trans_matrix_3],dim=-2)
    trans_matrix_1_3d = torch.cat([trans_matrix_1,trans_matrix_3],dim=-2)
    trans_matrix_2_3d = torch.cat([trans_matrix_2,trans_matrix_3],dim=-2) 

    Similarity_Matrix_3d_final = torch.matmul(Similarity_Matrix_3d,trans_matrix_1_3d)
    Similarity_Matrix_3d_final = torch.matmul(torch.inverse(trans_matrix_2_3d),Similarity_Matrix_3d_final)


    return Similarity_Matrix_3d_final