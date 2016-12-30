function [glints_x glints_y] = annotate_glints(dir_name, left_or_right)
% function [glints_x glints_y] = annotate_glints(dir_name, left_or_right)
% Annotate manully the glints in the images in folder 'dir_name'

colormap gray
if strcmp(left_or_right, 'left')
   ddir = dir([dir_name 'eyeImageLeft*']);
else
   ddir = dir([dir_name 'eyeImageRight*']);
end

n = 1;
for d=1:length(ddir)
   eval(sprintf('a = imread(''%s'');', [dir_name ddir(d).name]));
   clf, imagesc(a); axis off image,
   [gin_x gin_y button] = ginput(6);
   if any(button==3)
      disp('Skipping this image');
      continue;
   end
   glints_x(:,n) = gin_x;
   glints_y(:,n) = gin_y;
   
   fprintf('%d ', n);
   n = n+1;
end
fprintf('\n');
