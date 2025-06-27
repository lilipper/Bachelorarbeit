% Select a mode (Range profile=1; x_z_cut = 2; x_y_cut=3; 4= y_cut)
mode = 31;

if mode==1

    % Select range profile

    x_sample = 200;
    y_sample = 200;

    mag_z_final(:,:) = mag_final_all(:, x_sample, y_sample);
    
    % Normalize data (max of 3D dataset)

    z_final_image = mag_z_final./(max(max(max_mag_final)));

    %Show result

    plot(10*log10(z_final_image));
    set(gca, 'xLim', [600 800])
    set(gca, 'yLim', [-40 0])
    %     axis image;
    xlabel('Samples *dz [mm]');
    ylabel('Intensity norm. [dB]');
    % plot(1000*z_final_image);
    %     axis image;
    %     xlabel('Samples *dz [mm]');
    %     ylabel('Intensity norm. linear');



elseif mode==2

    % Select x_z_cut

    y_sample=230;

    mag_x_z_final(:,:)=mag_final_all(:,:,y_sample);

    % Normalize data (max of 3D dataset)

    x_z_final_image=mag_x_z_final./(max(max(max_mag_final)));

    [max_intens,ind]=max(x_z_final_image);

    imagesc(10*log10(x_z_final_image),[-50 0]);
    set(gca,'yLim',[690 710]);
    axis normal;
    colorbar;
    xlabel('Samples *dx [mm]');
    ylabel('Samples *dz [mm]');


elseif mode==3

    % Select x_y_cut
    % 702 is where most things can be seen with USAF
    % 699 for explosive target
    z_sample = 699;


    mag_x_y_final(:, :) = mag_final_all(z_sample, :, :);

    % Flip and rotate data

    ir1 = flipud(mag_x_y_final);
    x_y_final = rot90(ir1, 3);

    % Normalize data (max of 3D dataset)

    %     x_y_final_image=x_y_final./(max(max((mag_x_y_final))));
    x_y_final_image = x_y_final./max_mag_final_all;
    %     x_y_final_image=fliplr(x_y_final_image);

    cm = flipud(jet);
    colormap(cm);
    %     colormap(gray);

    %Show result
    x_y_final_image_log_scale = 10*log10(x_y_final_image);
    imagesc(10*log10(x_y_final_image), [-50 0]);
    im = gca;
    axis image;
    colorbar;
    set(im,'XDir','reverse');
    im.XTickLabel = x_min+(im.XTick .*dx);
    im.YTickLabel = y_min+(im.YTick .*dy);
    title(['XY Layer at z = ' num2str(z_sample)]);
    xlabel('x[mm]');
    ylabel('y[mm]');
    zlabel('Intensity normalized [dB]');
    % Saving image. Putting -1 to the index to make it the same as python
    % save_tiff(fliplr(x_y_final_image_log_scale), strcat('USAF_images/mat_log_image_z_', num2str(z_sample - 1), '.tiff'))

elseif mode==31

    % Select x_y_cut
    min_z = 695;
    for z_sample = min_z:(min_z + 3) %699:702

        mag_x_y_final(:, :) = mag_final_all(z_sample, :, :);

        % Flip and rotate data

        ir1 = flipud(mag_x_y_final);
        x_y_final = rot90(ir1, 3);

        % Normalize data (max of 3D dataset)

        x_y_final_image=x_y_final./(max(max(max_mag_final)));

        subplot(2, 2, z_sample - (min_z - 1));

        cm = flipud(jet);
        colormap(cm);

        %Show result

        imagesc(10*log10(x_y_final_image), [-35 0]);
        axis image;
        colorbar;
        title(['XY Layer at z = ' num2str(z_sample)]);
        xlabel('Samples *dx [mm]');
        ylabel('Samples *dy [mm]');
        zlabel('Intensity normalized [dB]');
    end
    % Flip Colormap


elseif mode==4

    % Select x_cut

    y_sample=250;
    z_sample=701;

    mag_x_z_final=mag_final_all(z_sample,:,y_sample);

    % Normalize data (max of 3D dataset)

    x_z_final_image=mag_x_z_final./(max(max(max_mag_final)));

    plot(10*log10(x_z_final_image));
    axis image;
    xlabel('Samples *dx [mm]');
    ylabel('Intensity norm. [dB]');

elseif mode==5

    % Select y_cut

    x_sample=238;
    z_sample=701;

    mag_y_z_final(:,:)=mag_final_all(z_sample,x_sample,:);

    % Normalize data (max of 3D dataset)

    y_z_final_image=mag_y_z_final./(max(max(max_mag_final)));

    plot(10*log10(y_z_final_image));
    axis image;
    xlabel('Samples *dx [mm]');
    ylabel('Intensity norm. [dB]');


elseif mode==6

    z_start=6900;
    z_stop=7060;



    for z_layer = z_start:z_stop

        mag_x_y_final(:,:)=mag_final_all(z_layer,:,:);

        % Flip and rotate data

        ir1=flipud(mag_x_y_final);
        x_y_final=rot90(ir1,3);

        % Normalize data (max of 3D dataset)

        x_y_final_image=x_y_final./max_mag_final_all;
        d_x_y_final_image=double(x_y_final_image);
        x_y_final_image_log=10*log10(d_x_y_final_image);


        %         imwrite(mat2gray(x_y_final_image_log),['z=' num2str(z_layer) '.tif']);
        imwrite(mat2gray(d_x_y_final_image),['z=' num2str(z_layer) '.tif']);

    end


elseif mode==7

    z_start=696;
    z_stop=719;

    % suplot specs
    row=4;
    column=6;

    for z_layer = z_start:z_stop

        mag_x_y_final(:,:)=mag_final_all(z_layer,:,:);

        % Flip and rotate data

        ir1=flipud(mag_x_y_final);
        x_y_final=rot90(ir1,3);

        % Normalize data (max of 3D dataset)

        x_y_final_image=x_y_final./max_mag_final_all;
        d_x_y_final_image=double(x_y_final_image);
        x_y_final_image_log=10*log10(d_x_y_final_image);

        z_num=z_layer-z_start+1
        figure
        %       ax1 = subplot(row,column,z_num);
        cm=flipud(jet);
        colormap(cm);

        %Color image

        imagesc(10*log10(x_y_final_image),[-50 0]);
        im=gca;
        axis image;
        colorbar;
        set(im,'XDir','reverse');
        im.XTickLabel=[x_min+(im.XTick .*dx)];
        im.YTickLabel=[y_min+(im.YTick .*dy)];
        title(['XY Layer at z=' num2str(z_layer)]);
        xlabel('x[mm]');
        ylabel('y[mm]');


        % Grayscale image

        %         imwrite(mat2gray(x_y_final_image_log),['z=' num2str(z_layer) '.tif']);
        filename=file_in(1:end-4);

        savefig(gcf,[filename ' z=' num2str(z_layer)],'compact');
        set(gcf,'PaperPositionMode','Auto','PaperUnits','centimeters','PaperSize',[21, 29.7],'units','centimeters','outerposition',[0 0 21 21]);
        saveas(gcf,[filename ' z=' num2str(z_layer)],'pdf')

    end
    % Maximize
    %     set(gcf, 'Position', get(0, 'Screensize'));

    % Save jpg
    %     filename=file_in(1:end-4);

    %     set(gcf,'Name',filename);
    %     orient('portrait');
    %     orient('landscape');

    %     saveas(gcf,filename,'jpg');
    %     print(filename,'-dpdf','-bestfit');
    % print(filename,'-dpdf','-fillpage');
end