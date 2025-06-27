% Data_Read
% Last edit (Editor/date): Onofre Martorell 25.03.2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comments:
% Final format: "data_complex" [NF--> Number of samples, NX--> Number of
% points per line, NY--> Number of lines]
% dx, dy: Spacing in x and y direction
% F_Start_meas --> Startfrequency [Hz]
% F_Stop_meas --> Stopfrequency [Hz]
clear all;

predata = 250; % From THz Setup (its not incorporated in the file header (yet))

postdata = 50; % From THz Setup ((its not incorporated in the file header (yet))

save_data = true;
%%%%%%%%%%%%%%%%%% get file name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[file_in, path_in, FilterIndex] = uigetfile('*.rdf', 'Please choose a file in rdf-format:');
if not(isstr(file_in))
    return
end
%%%% Filename save
name = split(file_in, ".");
name = name(1);
if save_data
    filename_mat = strcat(path_in, "/", name, ".mat");
    save(filename_mat, "predata", "postdata") %, "-v7.3"
end
tic
%%%%%%%%%%%%%%%%%% read file name %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid = fopen([path_in file_in], 'r', 'native');
line_str = '';
name = '';
ii = 0;
disp('### Loading File ...')

while not(strcmp(line_str,'end_of_header'))
    ii = ii + 1;
    line_str = fgetl(fid);
    if not(strcmp(line_str, 'end_of_header'))
        eval([char('header') char(num2str(ii)) '=line_str;']);
    end
end

while not(strcmp(name, 'end_of_var_list'))
    line_str = fgetl(fid);
    [name,value] = strread(line_str, '%s%f');

    if not(strcmp(name,'end_of_var_list'))
        eval([char(name) '=' char(num2str(value)) ';']);
        if save_data
            save(filename_mat, char(name), "-append")
        end
    end
end
toc;


tic
%%%%%%%%%%%%%%%%%% read data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Converting data ...')

data = fread(fid, 2*NX*NY*(NF + predata + postdata), 'single=>single');
fclose(fid);

data = reshape(data, 2, NX*NY*(NF + predata + postdata));
data_complex_all = complex(data(1, :), data(2, :));

clear data;

data_complex_all = reshape(data_complex_all, (NF + predata + postdata), NX, NY);
data_complex_all = data_complex_all((predata + 1):(NF + predata), :, :);

max_data_all_raw = max(abs(data_complex_all), [], "all");

if save_data
    disp('Saving ...')
    data_complex_all_1 = data_complex_all(1:700, :, :);
    data_complex_all_2 = data_complex_all(701:end, :, :);

    % Assuming 64 bits for a complex number and 1 bit = 1.25 × 10^-10 gigabytes
    data_size = numel(data_complex_all)*64 * 1.25 * 10^-10;
    num_cuts = ceil(data_size/2); % Max size allowed is 2GB
    len_cut = ceil(size(data_complex_all, 1)/num_cuts);

    % Cutting the data in chunks that can fit in a .mat file
    root_name = 'data_complex_all_';
    for ii = 1:num_cuts
        name = [root_name num2str(ii)];
        if ii < num_cuts
            tempTable = data_complex_all((ii - 1)*len_cut + 1:(ii)*len_cut, :, :);

        else
            tempTable = data_complex_all((ii - 1)*len_cut + 1:end, :, :);
        end
        eval([char(name) '= tempTable;']);
        save(filename_mat, name, "-append")
    end

    save(filename_mat, "root_name", "-append")
    save(filename_mat, "num_cuts", "-append")
    save(filename_mat, "len_cut", "-append")
end
toc;

