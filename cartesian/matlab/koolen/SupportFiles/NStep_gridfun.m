function [varargout]= gridfun(fun,varargin)
% GRIDFUN Apply a function to each combination input elements.
%   A = GRIDFUN(FUN, B, C, ...) applies the function specified by FUN to
%   each combination of elements of arrays B,C,... and returns the results
%   in array A. If the output of FUN is uniform A is matrix with
%   [I,J,...,N,M,...] = size(A) in which I=numel(B),J=numel(C),... and
%   [N,M,...] is the size of the output. If the output is not uniform A is
%   a cell array with [I,J,...]=size(A). The inputs B, C, ... can be
%   vectors or cell arrays.
%
%   Vectors or strings that have to be passed to the function FUN have to
%   be stored in a cell. For example, GRIDFUN(@sort,{b1,b2},1,{'descend'})
%   will sort vectors b1 and b2 in descending order.
%
%   A = GRIDFUN(FUN, B, C, ..., 'CellOutput') always outputs a cell array.
%
%   A = GRIDFUN(FUN, B, C, ..., 'ProgressIndication') shows during the
%   calculation a progress bar with an estimation of the time remaining.
%
%   A = GRIDFUN(FUN, B, C, ..., 'Squeeze') squeezes the output matrices.
%
%   A = GRIDFUN(FUN, B, C, ..., 'Transpose') transposes the output matrices.
%
%   Examples
%       function y = foo(x1,x2)
%       y = x1*x1'+x2;
%
%       gridfun(@foo,[2,3],[2,4])
%       ans =
%           6     8
%          11    13
%       gridfun(@foo,{[2,3]},[2,4])
%       ans =
%           15    17
%       gridfun(@foo,{[2,3],[4,5]},[2,4])
%       ans =
%           15    17
%           43    45
%       gridfun(@foo,{[2;3]},[2,4])
%       ans(:,:,1,1) =
%            6     8
%       ans(:,:,2,1) =
%            8    10
%       ans(:,:,1,2) =
%            8    10
%       ans(:,:,2,2) =
%           11    13
%       gridfun(@foo,{[2;3],[4,5]},[2,4])
%       ans = 
%           [2x2 double]    [2x2 double]
%           [        43]    [        45]
%    
%   This file is supplied as an addition to the draft paper:
%   "Analysis and Control of Legged Locomotion with Capture Points" 
%   - Part 2: Application to Three Simple Models -
%   submitted to IEEE TRO
%
%   For further information, contact:
%   Tomas de Boer, tomasdeboer@gmail.com, or    
%   Twan Koolen,   tkoolen@ihmc.us
%
%   Copyright 2010, Delft BioRobotics Laboratory
%   Daniël Karssen 
%   Delft University of Technology
%   $Revision: 1.0 $  $Date: February 2010 $

% options
celloutput = false; progressindication = false; 
squeezeoutput = false;  transposeoutput = false;% default settings
if any(strcmp(varargin,'CellOutput'))
    celloutput = true;
    varargin = varargin(~strcmp(varargin,'CellOutput'));
end
if any(strcmp(varargin,'ProgressIndication'))
    progressindication = true;
    varargin = varargin(~strcmp(varargin,'ProgressIndication'));
end
if any(strcmp(varargin,'Squeeze'))
    squeezeoutput = true;
    varargin = varargin(~strcmp(varargin,'Squeeze'));
end
if any(strcmp(varargin,'Transpose'))
    transposeoutput = true;
    varargin = varargin(~strcmp(varargin,'Transpose'));
end

% input and output sizes
Nin = length(varargin); % number of inputs
Nout = max(1,nargout); % number of outputs
for n=1:Nin, siz(n) = numel(varargin{n}); end % size inputs
I = prod(siz); % total number of grid points

% initialize variables
inputs = cell(Nin,1); % prealloct input cell array
varargout = cell(Nout,1); % prealloct output cell array
for n=1:Nout, varargout{n}=cell(I,1); end % prealloct output cell vector

% string for eval function 
func_str = '['; 
for n=1:Nout, func_str = [func_str, 'varargout{',int2str(n),'}{index} ']; end
func_str = [func_str, '] = fun(inputs{:});'];

if progressindication 
    bar_handle = waitbar(0,'Please wait...'); % create progress bar
    tic % start timer for progress indication
    counter=0;
    last_toc=0;
end 

%%% main loop
for index=randperm(I)%1:I
    gridpoint = index2gridpoint(index,siz); % select grid point
    % select input
    for n=1:Nin
        if iscell(varargin{n}) % select cell if input is cell array 
            inputs{n} = varargin{n}{gridpoint(n)}; 
        else % select element if input is  
            inputs{n} = varargin{n}(gridpoint(n));
        end
    end 
    eval(func_str) % eval function

    if progressindication % progress indication bar
        counter=counter+1;
        if toc-last_toc>1 % update maximal once every second
            last_toc = toc;
            progress(counter,I,last_toc,bar_handle)
        end
    end 
end

%%% reshape outputs
for n=1:Nout
    % if possible convert cell array into single matrix
    if ~celloutput
        % check if all outputs have the same size
        siz_output = cellfun(@size,varargout{n},'UniformOutput',0);
        if ~isempty(siz_output) && isequal(siz_output{:},siz_output{1})
            % create matrix
            dim = ndims(varargout{n}{1});
            varargout{n} = cat(dim+1,varargout{n}{:}); % create matrix A(N,M,...,I*J*...)
            varargout{n} = shiftdim(varargout{n},dim); % shift dim to A(I*J*...,N,M,...)
            varargout{n} = reshape(varargout{n},[siz,siz_output{1}]); % reshape to A(I,J,...,N,M,...)
        end
    end
    % reshape output cell vector into cell matrix
    if iscell(varargout{n}), varargout{n} = reshape(varargout{n},[siz,1]); end
    
    if squeezeoutput, varargout{n} = squeeze(varargout{n}); end 
    if transposeoutput, varargout{n} = varargout{n}.'; end 
end

if progressindication, close(bar_handle), end % close progress bar


%%% function to go from index to gridpoint
function [gridpoint] = index2gridpoint(index,dims)
gridpoint = zeros(length(dims),1);
k = [1 cumprod(dims(1:end-1))];
for i = length(dims):-1:1,
  vi = rem(index-1, k(i)) + 1;
  gridpoint(i) = (index - vi)/k(i) + 1;
  index = vi;     
end

%%% fuction for progress indication
function progress(index,I,time,bar_handle)
timetogo = time/index*(I-index)/(3600*24);
if timetogo<1
    etr_text = ['Estimated time remaining: ',datestr(timetogo,13)];
else
    etr_text = ['Estimated time remaining: ',int2str(fix(timetogo)), ' day(s) and ',datestr(rem(timetogo,1),13)];
end
totaltime = time/index*I/(3600*24);
if totaltime<1
    ett_text = ['Estimated total time: ',datestr(totaltime,13)];
else
    ett_text = ['Estimated total time: ',int2str(fix(totaltime)), ' day(s) and ',datestr(rem(totaltime,1),13)];
end
waitbar(index/I, bar_handle, {etr_text,ett_text});
