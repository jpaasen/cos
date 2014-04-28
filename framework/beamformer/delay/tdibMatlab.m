function image = tdibMatlab( mfdata, mtaxe, x_r, D, c, fc, xarr, yarr )
%TDIB_MATLAB time domain interpolation beamformer (simple for a single ping)
%
%  Nearfield time domain interpolation beamformer (also called: dynamic focused beamformer, 
%  Delay-and-Sum beamformer or backprojection algorithm)
%
%  function image = tdib_matlab( mfdata, mtaxe, x_r, D, c, fc, xarr, yarr )
%
%  INPUTS
%     mfdata   Matched filtered complex data (basebanded) [n_samples,n_hydros]
%     mtaxe    Matched filter time axis [n_samples,1] [s]
%     x_r      Rx position along body-x relative to the Tx position [n_hydros,1] [m] 
%     D        Rx distance i body-yz relative to the Tx position [n_hydros,1] [m]
%     c        Sound velocity [m/s]
%     fc       Center frequency [Hz]
%     xarr     X image position array (relative to the origin defined below) [n_y,n_x] [m]
%     yarr     Y image position array (relative to the origin defined below) [n_y,n_x] [m]
%
%  OUTPUTS
%     image    Dynamic focused image complex (single) [n_y,n_x]
%
%  This routine performs beamforming in Body coordinates. The origin of the beamforming is 
%  in the body xz-plane defined by the receiver-array and the transmitter (y is defined to
%  be zero). In this plane the origin is at the center of the PCA-array along x and z. 
%  See dynamic_focus_matlab.m for beamforming in Map coordinates (for SAS).
%
%  20071029 Hayden J Callow and Roy Edgar Hansen and Torstein Olsmo Saeboe
%  Part of FOCUS Synthetic Aperture Sonar Signal Processing Toolbox
%  Copyright 2007 FFI, Norway

%  Changelog:
%  23.05.08 Comments now match what actually happens - TSb
%  21.05.08 Inputs xarr and yarr changed to 2D and forced to be defined on input - TSb
%  20.11.07 Rewritten from z=0 geometry to "true slant-range" - TSb

mfdata = single(mfdata);
mtaxe = single(mtaxe);
x_r = single(x_r);
D = single(D);
c = single(c);
fc = single(fc);
xarr = single(xarr);
yarr = single(yarr);

% Check if mfdata is 3D. Multiple banks is not supported.
mfz = size( mfdata );
if length( mfz ) > 2
   focus_message( 'Several banks not supported. Only processing first bank' );
   mfdata = squeeze( mfdata(:,:,1) );
   mfz = size( mfdata );
end

% Estimate the sizes
n_hydros = mfz(2);
n_x = size( xarr, 2 );
n_y = size( yarr, 1 );

% Define some useful variables
jtopifc = j*2*pi*fc;
inv_c   = 1./c;

% Choose center of PCA as origin in x and center of PCA as origin in z
xx = mean(x_r)/2;
zz = mean(D)/2;

% For all beams 
image = zeros(n_y,n_x);
for nx = 1:n_x
   
   % Range from transmitter to slant-range beam
   r_t = sqrt( ( xx + xarr(:,nx) ).^2 + yarr(:,nx).^2 + zz.^2 );
   
   % For all hydrophones
   for n=1:n_hydros
      
      % Range from slant-range beam to receiver
      r_r = sqrt( ( xx + xarr(:,nx) - x_r(n) ).^2 + yarr(:,nx).^2 + ( zz - D(n) ).^2 );

      % Travel time for all pixels in this beam and this hydrophome
      tipos = ( ( r_t + r_r ) * inv_c ); 

      % Select ping data
      tmp = single( mfdata(:,n) );
      
      % Interpolate into correct time for all pixels in this beam and this hydrophone
      tmp2 = interp1( mtaxe, tmp, tipos, 'linear' );
      
%      if( n==31 )
%         fprintf('!');
%      end
      
      % Mix to carrier
      tmp3 = tmp2 .* exp( jtopifc .* tipos );
      
      % Sum with all the other hydrophones
      image(:,nx) = image(:,nx) + tmp3;
      
   end

end

% Scale output with the number of hydrophones
image = single( image ) .* (1/n_hydros);

% End of file tdib_matlab.m