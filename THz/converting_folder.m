% === Pfad zum Ordner mit .rdf-Dateien (hier anpassen!) ===
data_folder = 'C:\Users\lgrli\Bachelorarbeit_VSC_Workspace\Bachelorarbeit\700GHz\700GHz';
disp('=== Starte Konvertierung aller RDF-Dateien ===')
% Liste aller .rdf-Dateien im angegebenen Ordner
rdf_files = dir(fullfile(data_folder, '*.rdf'));

% Konfigurationsparameter
predata = 250;
postdata = 50;
save_data = true;

% === Hauptschleife über alle RDF-Dateien ===
for k = 1:length(rdf_files)
    file_in = rdf_files(k).name;
    path_in = rdf_files(k).folder;
    full_path = fullfile(path_in, file_in);

    fprintf('Verarbeite Datei %s ...\n', file_in);

    % Ziel-Dateiname (.mat)
    name_parts = split(file_in, '.');
    name = name_parts{1};
    filename_mat = fullfile(path_in, [name, '.mat']);

    if save_data
        save(filename_mat, 'predata', 'postdata');
    end

    fid = fopen(full_path, 'r', 'native');
    if fid == -1
        warning('Datei %s konnte nicht geöffnet werden. Überspringe.', file_in);
        continue;
    end

    % Header lesen
    line_str = '';
    ii = 0;
    while ~strcmp(line_str, 'end_of_header')
        ii = ii + 1;
        line_str = fgetl(fid);
        if ~strcmp(line_str, 'end_of_header')
            eval(['header' num2str(ii) ' = line_str;']);
        end
    end

    % Variablen lesen
    name_var = '';
    while ~strcmp(name_var, 'end_of_var_list')
        line_str = fgetl(fid);
        tokens = textscan(line_str, '%s %f');
        if ~isempty(tokens{1})
            name_var = tokens{1}{1};
            if ~strcmp(name_var, 'end_of_var_list')
                value = tokens{2};
                eval([name_var ' = ' num2str(value) ';']);
                if save_data
                    save(filename_mat, name_var, '-append');
                end
            end
        end
    end

    % Daten lesen
    data = fread(fid, 2 * NX * NY * (NF + predata + postdata), 'single=>single');
    fclose(fid);

    data = reshape(data, 2, NX * NY * (NF + predata + postdata));
    data_complex_all = complex(data(1, :), data(2, :));
    data_complex_all = reshape(data_complex_all, (NF + predata + postdata), NX, NY);
    data_complex_all = data_complex_all((predata + 1):(NF + predata), :, :);

    % Daten chunkweise speichern
    data_size = numel(data_complex_all) * 64 * 1.25e-10; % in GB
    num_cuts = ceil(data_size / 2);
    len_cut = ceil(size(data_complex_all, 1) / num_cuts);

    root_name = 'data_complex_all_';
    for ii = 1:num_cuts
        var_name = [root_name, num2str(ii)];
        if ii < num_cuts
            temp = data_complex_all((ii - 1)*len_cut + 1 : ii*len_cut, :, :);
        else
            temp = data_complex_all((ii - 1)*len_cut + 1 : end, :, :);
        end
        eval([var_name ' = temp;']);
        save(filename_mat, var_name, '-append');
    end

    save(filename_mat, 'root_name', 'num_cuts', 'len_cut', '-append');
    disp(['Fertig: ', filename_mat]);
end
