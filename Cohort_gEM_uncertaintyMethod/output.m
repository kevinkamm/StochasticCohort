function output(fileName,...
                N,M,...
                ctimeCal,ctimeCalQ,...
                errCal,errCalQ,...
                paramTable,h,...
                mDCgan,sDDgan,dMLgan,iRSgan,rSOgan,...
                mDCcal,sDDcal,dMLcal,iRScal,rSOcal,...
                mDCcalQ,sDDcalQ,dMLcalQ,iRScalQ,rSOcalQ,...
                figRDP,figRDQ,figTraP,figTraQ,...
                varargin)
backgroundColor='w';
textColor='k';
compileLatex = true;
for iV=1:2:length(varargin)
    switch varargin{iV}
        case 'backgroundColor'
            backgroundColor = varargin{iV+1};
        case 'textColor'
            textColor = varargin{iV+1};
        case 'compileLatex'
            compileLatex = varargin{iV+1};
    end
end
fclose('all');
picType='eps';
saveParam='epsc';
numDict={'One','Two','Three','Four','Five','Six','Seven','Eight','Nine'};

root=[pwd, '/' ,'Results'];
pdfRoot=[root,'/','Pdf'];
tempPath=[pdfRoot,'/','temp'];
copyPath=[pdfRoot,'/',fileName];
templatePath=[tempPath,'/','template', '.','tex'];
if compileLatex
    outputFilePath={[copyPath,'/','template','.','pdf'],...
                [copyPath,'/','template','.','tex']};
    copyFilePath={[copyPath,'/',fileName,'.','pdf'],...
                [copyPath,'/',fileName,'.','tex']};
else
    outputFilePath={[copyPath,'/','template','.','tex']};
    copyFilePath={[copyPath,'/',fileName,'.','tex']};
end
inputPath=[tempPath,'/','input','.','tex'];

% Delete auxiliary files in temp folder
try
cleanDir(tempPath,{'template.tex'});
catch
    disp('Error in cleandir')
end
delDir(copyPath)
mkDir(copyPath)

inputFile=fopen(inputPath,'w+');
%% Head
fprintf(inputFile,...
        '\\section{Rating transition model}\n');
%% Calibration Paramters
fprintf(inputFile,...
        '\\subsection{Calibration Parameters}\n');
[latexFilePath,latexCommand]=saveCalibrationParameters('CalibrationParam');
fprintf(inputFile,...
        '\t\\input{%s}\n',changeSlash(latexFilePath));
for iLC=1:1:length(latexCommand)
    fprintf(inputFile,...
            '\t%s\n\n\n',latexCommand{iLC});
end
%% Computational Times
fprintf(inputFile,...
        '\\subsection{Computational Times}\n');
[latexFilePath,latexCommand]=saveComputationalTimes('CompTimes');
fprintf(inputFile,...
        '\t\\input{%s}\n',changeSlash(latexFilePath));
for iLC=1:1:length(latexCommand)
    fprintf(inputFile,...
            '\t%s\n\n\n',latexCommand{iLC});
end
%% Errors
fprintf(inputFile,...
        '\\subsection{Errors}\n');
[latexFilePath,latexCommand]=saveErrors('Errors');
fprintf(inputFile,...
        '\t\\input{%s}\n',changeSlash(latexFilePath));
for iLC=1:1:length(latexCommand)
    fprintf(inputFile,...
            '\t%s\n\n\n',latexCommand{iLC});
end

%% Rating properties
% GAN
fprintf(inputFile,...
        '\\subsection{Rating Properties}\n');
fprintf(inputFile,...
        '\\paragraph*{TimeGAN}\n');
[latexFilePath,latexCommand]=saveRatingProperties(mDCgan,sDDgan,dMLgan,iRSgan,rSOgan,'Rec','ratingProperties');
fprintf(inputFile,...
        '\t\\input{%s}\n',changeSlash(latexFilePath));
for iLC=1:1:length(latexCommand)
    fprintf(inputFile,...
            '\t%s\n\n\n',latexCommand{iLC});
end
fprintf(inputFile,...
        '\\paragraph*{SDE under P}\n');
% SDE P
[latexFilePath,latexCommand]=saveRatingProperties(mDCcal,sDDcal,dMLcal,iRScal,rSOcal,'SDEP','ratingPropertiesP');
fprintf(inputFile,...
        '\t\\input{%s}\n',changeSlash(latexFilePath));
for iLC=1:1:length(latexCommand)
    fprintf(inputFile,...
            '\t%s\n\n\n',latexCommand{iLC});
end
fprintf(inputFile,...
        '\\paragraph*{SDE under Q}\n');
% SDE Q
[latexFilePath,latexCommand]=saveRatingProperties(mDCcalQ,sDDcalQ,dMLcalQ,iRScalQ,rSOcalQ,'SDEQ','ratingPropertiesQ');
fprintf(inputFile,...
        '\t\\input{%s}\n',changeSlash(latexFilePath));
for iLC=1:1:length(latexCommand)
    fprintf(inputFile,...
            '\t%s\n\n\n',latexCommand{iLC});
end
%% Plots of Distributions
if ~isempty(figRDP)
    fprintf(inputFile,...
            '\\subsection{Plots of Distributions}\n');
    latexFilePath=saveFigures(figRDP,'RD_P',['RD_P']);
    fprintf(inputFile,...
            '\t\\input{%s}\n',changeSlash(latexFilePath));
end
if ~isempty(figRDQ)
    fprintf(inputFile,...
            '\\subsection{Plots of Distributions}\n');
    latexFilePath=saveFigures(figRDQ,'RD_Q',['RD_Q']);
    fprintf(inputFile,...
            '\t\\input{%s}\n',changeSlash(latexFilePath));
end
%% Plots of Trajectories
if ~isempty(figTraP)
    fprintf(inputFile,...
            '\\subsection{Plots of Trajectories}\n');
    latexFilePath=saveFigures(figTraP,'Tra_P',['Tra_P']);
    fprintf(inputFile,...
            '\t\\input{%s}\n',changeSlash(latexFilePath));
end
if ~isempty(figTraQ)
    fprintf(inputFile,...
            '\\subsection{Plots of Trajectories}\n');
    latexFilePath=saveFigures(figTraQ,'Tra_Q',['Tra_Q']);
    fprintf(inputFile,...
            '\t\\input{%s}\n',changeSlash(latexFilePath));
end
%% Close file
fclose(inputFile);
%% Compile Latex
if compileLatex
currFolder=cd(tempPath);
str1=sprintf('pdflatex %s',templatePath);
% system(str1)
[returncode, ~] = system(str1);
cd(currFolder);
end

copyfile(tempPath,copyPath);
% Renaming files
for iFile=1:1:length(copyFilePath)
    movefile(outputFilePath{iFile},copyFilePath{iFile})
end
% Delete auxiliary latex files in copy folder
delete([copyPath,'/','template*.*']);
% Delete auxiliary files in temp folder
% cleanDir(tempPath,{'template.tex'});
fclose('all');
%% Latex functions
    function [latexFilePath,latexCommand]=saveRatingProperties(mDC,sDD,dML,iRS,rSO,method,saveAt)
            latexCommand={};
            if strcmp(saveAt, '')
                latexFilePath=['ratingProperties',method,'.tex'];
            else
                latexFilePath=[saveAt,'/','ratingProperties',method,'.tex'];
                mkDir([tempPath,'/',saveAt]);
            end
            file = fopen([tempPath,'/',latexFilePath],'w+');

            latexCommand{end+1}=['\mDC',method];
            fprintf(file,'\\newcommand{%s}{\n',latexCommand{end});
            fprintf(file,'Monotone default column\\\\\n');
            columnNames=mDC.Properties.VariableNames;
            rowNames=mDC.Properties.RowNames;
            fprintf(file,'\\begin{tabular}{*{%d}{c}}\n',length(columnNames)+1);
            fprintf(file,...
                mat2Table(' ',columnNames,rowNames,...
                mDC.Variables,{'','%1.2e','',''},{'','','',''},...
                'headHook','\\toprule'));
            fprintf(file,'\\end{tabular}\n');
            fprintf(file,'}\n');

            latexCommand{end+1}=['\sDD',method];
            fprintf(file,'\\newcommand{%s}{\n',latexCommand{end});
            fprintf(file,'Strongly diagonal dominant\\\\\n');
            columnNames=sDD.Properties.VariableNames;
            rowNames=sDD.Properties.RowNames;
            fprintf(file,'\\begin{tabular}{*{%d}{c}}\n',length(columnNames)+1);
            fprintf(file,...
                mat2Table(' ',columnNames,rowNames,...
                sDD.Variables,{'','%1.2e','',''},{'','','',''},...
                'headHook','\\toprule'));
            fprintf(file,'\\end{tabular}\n');
            fprintf(file,'}\n');

            latexCommand{end+1}=['\dML',method];
            fprintf(file,'\\newcommand{%s}{\n',latexCommand{end});
            fprintf(file,'Down more likely\\\\\n');
            columnNames=dML.Properties.VariableNames;
            rowNames=dML.Properties.RowNames;
            fprintf(file,'\\begin{tabular}{*{%d}{c}}\n',length(columnNames)+1);
            fprintf(file,...
                mat2Table('',columnNames,rowNames,...
                dML.Variables,{'','%1.2e','',''},{'','','',''},...
                'headHook','\\toprule'));
            fprintf(file,'\\end{tabular}\n');
            fprintf(file,'}\n');
            
            latexCommand{end+1}=['\iRS',method];
            fprintf(file,'\\newcommand{%s}{\n',latexCommand{end});
            fprintf(file,'Increasing rating spread\\\\\n');
            columnNames=iRS.Properties.VariableNames;
            rowNames=iRS.Properties.RowNames;
            fprintf(file,'\\begin{tabular}{*{%d}{c}}\n',length(columnNames)+1);
            fprintf(file,...
                mat2Table(' ',columnNames,rowNames,...
                iRS.Variables,{'','%1.2e','',''},{'','','',''},...
                'headHook','\\toprule'));
            fprintf(file,'\\end{tabular}\n');
            fprintf(file,'}\n');
            
            latexCommand{end+1}=['\rSO',method];
            fprintf(file,'\\newcommand{%s}{\n',latexCommand{end});
            fprintf(file,'Row sum one\\\\\n');
            columnNames=rSO.Properties.VariableNames;
            rowNames=rSO.Properties.RowNames;
            fprintf(file,'\\begin{tabular}{*{%d}{c}}\n',length(columnNames)+1);
            fprintf(file,...
                mat2Table(' ',columnNames,rowNames,...
                rSO.Variables,{'','%1.2e','',''},{'','','',''},...
                'headHook','\\toprule'));
            fprintf(file,'\\end{tabular}\n');
            fprintf(file,'}\n');

            fclose(file);
    end
    function [latexFilePath,latexCommand]=saveCalibrationParameters(saveAt)
            latexCommand={};
            if strcmp(saveAt, '')
                latexFilePath='calParams.tex';
            else
                latexFilePath=[saveAt,'/','calParams.tex'];
                mkDir([tempPath,'/',saveAt]);
            end
            file = fopen([tempPath,'/',latexFilePath],'w+');
            latexCommand{end+1}=['\calibrationParametersP'];
            fprintf(file,'\\newcommand{%s}{\n',latexCommand{end});
            columnNames=paramTable.Properties.VariableNames;
            rowNames=paramTable.Properties.RowNames;
            fprintf(file,'\\begin{tabular}{*{%d}{c}}\n',length(columnNames)+1);
            fprintf(file,...
                mat2Table('From-To',columnNames,rowNames,...
                paramTable.Variables,{'','%1.2e','',''},{'','','',''},...
                'headHook','\\toprule'));
            fprintf(file,'\\end{tabular}\n');
            fprintf(file,'}\n');

            latexCommand{end+1}=['\calibrationParametersQ'];
            fprintf(file,'\\newcommand{%s}{\n',latexCommand{end});
            fprintf(file,['$h=\\left[',sprintf('%1.3f\\\\ ',h),'\\right]$']);
            fprintf(file,'}\n');
            fclose(file);
    end

    function [latexFilePath,latexCommand]=saveComputationalTimes(saveAt)
        latexCommand={'\compTimes'};
        if strcmp(saveAt, '')
            latexFilePath='compTimes.tex';
        else
            latexFilePath=[saveAt,'/','compTimes.tex'];
            mkDir([tempPath,'/',saveAt]);
        end
        file = fopen([tempPath,'/',latexFilePath],'w+');
        fprintf(file,...
                '\\newcommand{\\ctimeCalP}{%g}\n',ctimeCal);
        fprintf(file,...
                '\\newcommand{\\ctimeCalQ}{%g}\n',ctimeCalQ);
        fprintf(file,...
                '\\newcommand{%s}{\n',latexCommand{1});
        fprintf(file,...
                '\\begin{compactenum}\n');
        fprintf(file,...
                '\\item Calibration time under P using %d points for the Euler scheme and %d trajectories: \\ctimeCalP seconds.\n',N,M);
        fprintf(file,...
                '\\item Calibration time under Q using %d points for the Euler scheme and %d trajectories: \\ctimeCalQ seconds.\n',N,M);
        fprintf(file,...
                '\\end{compactenum}\n');
        fprintf(file,...
                '}\n');    
        fclose(file);
    end

    function [latexFilePath,latexCommand]=saveErrors(saveAt)
        latexCommand={'\errors'};
        if strcmp(saveAt, '')
            latexFilePath='errors.tex';
        else
            latexFilePath=[saveAt,'/','errors.tex'];
            mkDir([tempPath,'/',saveAt]);
        end
        file = fopen([tempPath,'/',latexFilePath],'w+');
        fprintf(file,...
                '\\newcommand{\\errorCalP}{%g}\n',errCal);
        fprintf(file,...
                '\\newcommand{\\errorCalQ}{%g}\n',errCalQ);
        fprintf(file,...
                '\\newcommand{%s}{\n',latexCommand{1});
        fprintf(file,...
                '\\begin{compactenum}\n');
        fprintf(file,...
                '\\item Calibration error under P using %d points for the Euler scheme and %d trajectories: \\errorCalP.\n',N,M);
        fprintf(file,...
                '\\item Calibration error under Q using %d points for the Euler scheme and %d trajectories: \\errorCalQ.\n',N,M);
        fprintf(file,...
                '\\end{compactenum}\n');
        fprintf(file,...
                '}\n');    
        fclose(file);
    end

    function latexFilePath=saveFiguresLandscape(figures,saveAt,figName)
        if strcmp(figName,'')
            figName='fig';
        end
        if strcmp(saveAt, '')
            latexFilePath='figure.tex';
            latexFolderPath=tempPath;
        else
            latexFilePath=[saveAt,'/','figure.tex'];
            latexFolderPath=[tempPath,'/',saveAt];
            mkDir([tempPath,'/',saveAt]);
        end
        file = fopen([tempPath,'/',latexFilePath],'w+');
        for i=1:1:length(figures)
            picName=sprintf('%s_%d',figName,i);
            picPath = [latexFolderPath,'/',picName,'.',picType];
            if strcmp(saveAt, '')
                relPicPath=picName;
            else
                relPicPath=[saveAt,'/',picName];
            end
            if iscell(figures)
                exportgraphics(figures{i},picPath,'BackgroundColor',backgroundColor)
            else
                exportgraphics(figures(i),picPath,'BackgroundColor',backgroundColor)
            end
            fprintf(file,...
                '\\begin{landscape}\n');
            fprintf(file,...
                '\\includegraphics[width=.95\\columnwidth]{%s}\n',...
                changeSlash(relPicPath));
            fprintf(file,...
                '\\end{landscape}\n');
        end
        fclose(file);
    end
    function latexFilePath=saveFigures(figures,saveAt,figName)
        if strcmp(figName,'')
            figName='fig';
        end
        if strcmp(saveAt, '')
            latexFilePath='figure.tex';
            latexFolderPath=tempPath;
        else
            latexFilePath=[saveAt,'/','figure.tex'];
            latexFolderPath=[tempPath,'/',saveAt];
            mkDir([tempPath,'/',saveAt]);
        end
        file = fopen([tempPath,'/',latexFilePath],'w+');
        for i=1:1:length(figures)
            picName=sprintf('%s_%d',figName,i);
            picPath = [latexFolderPath,'/',picName,'.',picType];
            if strcmp(saveAt, '')
                relPicPath=picName;
            else
                relPicPath=[saveAt,'/',picName];
            end
            if iscell(figures)
                exportgraphics(figures{i},picPath,'BackgroundColor',backgroundColor)
            else
                exportgraphics(figures(i),picPath,'BackgroundColor',backgroundColor)
            end
            fprintf(file,...
                '\\begin{minipage}[c][][c]{\\linewidth}\n');
            fprintf(file,...
                '\\includegraphics[width=.95\\columnwidth]{%s}\n',...
                changeSlash(relPicPath));
            fprintf(file,...
                '\\end{minipage}\n');
        end
        fclose(file);
    end
    function latexFilePath=saveVideos(videos,saveAt)
        if strcmp(saveAt, '')
            latexFilePath='video.tex';
            latexFolderPath=tempPath;
        else
            latexFilePath=[saveAt,'/','video.tex'];
            latexFolderPath=[tempPath,'/',saveAt];
            mkDir([tempPath,'/',saveAt]);
        end
        file = fopen([tempPath,'/',latexFilePath],'w+');
        for i=1:1:length(videos)
            temp = strsplit(videos{i},'/');
            currVidName = [temp{end},'.pdf'];
            currVid = [latexFolderPath,'/',currVidName];
            copyfile([videos{i},'.pdf'],currVid);
            if strcmp(saveAt, '')
                relVidPath=currVidName;
            else
                relVidPath=[saveAt,'/',currVidName];
            end
            fprintf(file,...
                '\\begin{minipage}[c][][c]{\\linewidth}\n');
            fprintf(file,...
                '\\animategraphics[width=\\linewidth,autoplay,controls]{12}{%s}{}{}\n',...
                changeSlash(replace(relVidPath,'.pdf','')));
            fprintf(file,...
                '\\end{minipage}\n');
        end
        fclose(file);
    end

end
%% Auxiliary functions
function str=changeSlash(str)
    for i=1:1:length(str)
        if strcmp(str(i),'/')
            str(i)='/';
        end
    end
end
function delFile(file)
    if exist(file)
        delete(file);
    end
end
function delDir(dir)
    if exist(dir)==7
        rmdir(dir,'s');
    end
end
function cleanDir(mdir,except)
    except{end+1}='.';
    except{end+1}='..';
    for d = dir(mdir).'
      if ~any(strcmp(d.name,except))
          if d.isdir
              rmdir([d.folder,'/',d.name],'s');
          else
              delete([d.folder,'/',d.name]);
          end
      end
    end
end
function mkDir(dir)
    if exist(dir)==0
        mkdir(dir);
    end
end
function latexStr=mat2Table(corner,head,index,body,formatSpec,unit,varargin)
%     percent='\\,\\%%';
% formatSpec, unit order: index, body, head, corner
    headHook='';
    bodyHook='';
    for k=1:1:length(varargin)
        switch varargin{k}
            case 'headHook'
                headHook=varargin{k+1};
            case 'bodyHook'
                bodyHook=varargin{k+1};
        end
    end
    latexStr='';
    function x=functor(x,formatSpec,unit)
        if ~ischar(x)
            if mod(abs(x),1)==0 
                x=num2str(x,'%d');
            else
                x=num2str(x,formatSpec);
            end
        end
        x=[x,unit];
    end
    if ~isempty(corner)
        cornerStr=functor(corner,formatSpec{4},unit{4});
        latexStr=[latexStr,cornerStr,' & '];
    end
    if ~isempty(head)
        if iscell(head)
            headCell=cellfun(@(x)functor(x,formatSpec{3},unit{3}),head,...
                'UniformOutput',false);
        else
            headCell=arrayfun(@(x)functor(x,formatSpec{3},unit{3}),head,...
                'UniformOutput',false);
        end
        headStr=join(headCell,' & ');
        latexStr=[latexStr,headStr{1},'\\\\\n'];
    end
    latexStr=[latexStr,headHook];
    bodyCell=arrayfun(@(x)functor(x,formatSpec{2},unit{2}),body,...
        'UniformOutput',false);
    if ~isempty(index)
        if iscell(index)
            indexCell=cellfun(@(x)functor(x,formatSpec{1},unit{1}),index,...
                'UniformOutput',false);
        else
            indexCell=arrayfun(@(x)functor(x,formatSpec{1},unit{1}),index,...
                'UniformOutput',false);
        end
        bodyCell=cat(2,indexCell,bodyCell);
    end
    bodyStr=join(join(bodyCell,' & ',2),'\\\\\n',1);
    latexStr=[latexStr,bodyStr{1}];
    latexStr=[latexStr,bodyHook];
end