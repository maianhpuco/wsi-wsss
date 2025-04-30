vqgan_logs_dir = '/project/hnguyen2/mvu9/folder_04_ma/wsi_efficient_seg/resources/vqgan/logs'
is_gumbel = True 

batch_size = 16
num_epochs = 50
learning_rate = 1e-4 


if dataset_name == 'bcss': 
    datastet_dir ="/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"
elif dataset_name == 'luad' 
    datastet_dir =  "/project/hnguyen2/mvu9/datasets/processing_datasets/LUAD-HistoSeg_organized"   
else:
    raise ValueError("Invalid dataset name. Choose either 'bcss' or 'luad'.")  