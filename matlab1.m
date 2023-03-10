addpath('D:\FYP');
clc; clear all;
dataset = 'B';
dataset_name = ['shanghaitech_part_' dataset ];
path = ['D:\FYP\MCNN\data/original/shanghaitech/part_' dataset '_final/test_data/images/'];
gt_path = ['D:\FYP\MCNN\data/original/shanghaitech/part_' dataset '_final/test_data/ground_truth/'];
gt_path_csv = ['D:\FYP\MCNN\data/original/shanghaitech/part_' dataset '_final/test_data/ground_truth_csv/'];

mkdir(gt_path_csv )
if (dataset == 'A')
    num_images = 182;
else
    num_images = 316;
end

for i = 1:num_images    
    fprintf(1,'Processing %3d/%d files\n', i, num_images);
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end     
    annPoints =  image_info{1}.location;   
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints);    
    csvwrite([gt_path_csv ,'IMG_',num2str(i) '.csv'], im_density);       
end