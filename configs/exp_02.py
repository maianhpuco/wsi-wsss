vqgan_logs_dir = '/project/hnguyen2/mvu9/folder_04_ma/wsi_efficient_seg/resources/vqgan/logs'
dataset_name = 'bcss'

if dataset_name == 'bcss': 
    datastet_dir ="/project/hnguyen2/mvu9/datasets/processing_datasets/BCSS-WSSS_organized"
elif dataset_name == 'luad' 
    datastet_dir =  "/project/hnguyen2/mvu9/datasets/processing_datasets/LUAD-HistoSeg_organized"   
else:
    raise ValueError("Invalid dataset name. Choose either 'bcss' or 'luad'.")  