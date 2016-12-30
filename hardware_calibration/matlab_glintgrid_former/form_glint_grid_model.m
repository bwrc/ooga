% Form the glint model
% GLINTSx and GLINTSy contain the annotated glint coordinates

Ng = size(GLINTSx,1);  % number of glints
GLINTSx_m = GLINTSx - repmat(mean(GLINTSx), [Ng 1]);
GLINTSy_m = GLINTSy - repmat(mean(GLINTSy), [Ng 1]);

scales = sqrt(var(GLINTSx_m) + var(GLINTSy_m));
GLINTSx_m_s = GLINTSx_m ./ repmat(scales, [Ng 1]);
GLINTSy_m_s = GLINTSy_m ./ repmat(scales, [Ng 1]);

MUx = mean(GLINTSx_m_s');
MUy = mean(GLINTSy_m_s');    
C = cov([GLINTSx_m_s' GLINTSy_m_s']);
mean_scale = mean(scales);    
