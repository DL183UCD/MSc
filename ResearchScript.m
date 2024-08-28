%Path to comparison file
function ResearchScript(compareFile)
    % This script performs 
    % 1. pairwise comparisons of average N rates by t-tests, 
    % 2. Generates barcharts illustrating the average N rates
    % 3. pairwise comparison of flow rates by t-tests at each time point
    % 4. Ribbon plot of flow rates
    % 5. Simulated samples; Repeating 1-4 for simulated samples.
    % Input: 
    %  compareFile     | string | Path to xlsx comparison file

    %compareFile = "Research Output/comparison_test2.xlsx";
    
    [filePath, fileName] = fileparts(compareFile);
    
    %Load data from xlsx comparison file
    [flows, cumFlows, stdFlows, ...
    avgFlows, pools,stdPools, ...
    observations, poolNames, ...
    treatments, measuredTimes, ...
    alpha, pAdjustMethod,...
    transformations, transformationMatrix] = load_from_file(compareFile);
    
    nSamples = nan(length(treatments),1);
    for i = 1:length(treatments)
        nSamples(i) = numel(observations(:,:,:,i));
    end
    testsToConduct = nchoosek(length(treatments),2);
    %-------- PERFORM TESTS ON AVEREAGE N RATES ---------%
    %Create Bar chart illustrating average flow rates
    %Bar charts of avg N rates
    tfig = tiledlayout(ceil(length(transformations)/4),4)
    for i = 1:length(transformations)
        nexttile;
        create_bar_Chart(avgFlows(:,i), stdFlows(:,i), treatments, transformations{i});
    end
    saveas(tfig,filePath+"/"+"Modelled_Avg_N_Rates_Bar_Chart","fig");
    %pause;
    %Perform pairwise t-tests on average flow rates
    avgNtest = struct;
    allPvalues = zeros(testsToConduct * length(transformations),1);
    indxP = 1;
    for i=1:length(transformations)
        avgNtest.(transformations{i}) = create_results_table(treatments, avgFlows(:,i), stdFlows(:,i), nSamples, alpha);
        allPvalues(indxP:(i*testsToConduct)) = avgNtest.(transformations{i}).pValue;
        indxP = indxP + testsToConduct;
    end
    %Perform pvalue adjustment
    allPvalues = pval_adjust(allPvalues, pAdjustMethod);
    indxP = 1;
    %Adjust pvalues in all tables
    for i = 1:length(transformations)
        avgNtest.(transformations{i}).pValue = allPvalues(indxP:(i*testsToConduct));
        indxP = indxP + testsToConduct;
    end
    %Write to excel file
    write_to_excel(avgNtest,tfig, filePath+"/"+"Modelled_Avg_N_rates.xlsx");
    %-------- PERFORM TESTS ON FLOW RATES ---------%
    %Plots of flow rates
    tfig = tiledlayout(ceil(length(transformations)/4),4)
    for i = 1:length(transformations)
        nexttile;
        create_flow_graph(squeeze(flows(:,i,:)), stdFlows(:,i), treatments, alpha, "Plot of instantaneous Flow "+ transformations{i});
    end
    saveas(tfig, filePath+"/"+"Modelled_Flow_Rates","fig");
    %pause;
    %T-test plots (pairwise comparison of flow rate at each time point)
    %Conduct t-test on each pairwise time point
    %Set threshold to alpha, and see what time point's graph is above alpha

    flowNTest = struct;
    tableColumnNames = cell(measuredTimes(end)+2,1);
    tableColumnNames{1} = 'Group1';
    tableColumnNames{2} = 'Group2';
    for i = 1:measuredTimes(end)
        tableColumnNames{2+i} = strcat('P(t=',int2str(i),')');
    end
    for i = 1:length(transformations)
        out = cell(testsToConduct,measuredTimes(end)+2);
        tfig = tiledlayout(ceil(testsToConduct / 2), 4)
        indx = 1;
        for j = 1:length(treatments)
            for k = (j+1):length(treatments)
                nexttile
                create_flow_graph(squeeze(flows(:,i,[j k])), stdFlows([j k],i), treatments([j k]), alpha, "Plot of Flow Rate "+ transformations{i});
                nexttile;
                [~, ~, ~, p] = independent_t_test(flows(:,i,j), flows(:,i,k), stdFlows(j, i), stdFlows(k, i), nSamples(j), nSamples(k), alpha);
                %perform pvalue adjustment
                p = pval_adjust(p, "none");
                yline(alpha, '-','Threshold')
                xlabel("Time");
                ylabel("P value");
                title(strcat("Probability of Equal Rate at Time T"));
                hold on
                scatter(1:length(p), p, "Marker", "x");
                hold off
                %Append to table
                out{indx, 1} = treatments{k};
                out{indx, 2} = treatments{j};
                for z = 2:(measuredTimes(end)+2)
                    out{indx, z} = p(z-1);
                end
                indx = indx + 1;
            end
            saveas(tfig, filePath+"/"+"Modelled_Flow_Rate_"+transformations{i},"fig");
        end
        flowNTest.(transformations{i}) = cell2table(out, 'VariableNames', tableColumnNames);
        %pause;

        writetable(flowNTest.(transformations{i}), filePath+"/"+"Modelled_Flow_Rates.xlsx", 'Sheet', transformations{i}, 'Range', 'A1');
        %Extract figure from tiled layout and paste to excel sheet
        try 
            matlab.io.internal.getExcelInstance;
            for j = 1:testsToConduct
                if ishandle(tfig)
    
                    ax = nexttile(2*j-1);
                    fig = copyobj(ax,figure);
                    set(fig,'OuterPosition',[0 0 1 1]);
                    xlswritefig(fig, filePath+"/"+"Modelled_Flow_Rates.xlsx", transformations{i}, strcat('A', int2str(testsToConduct+j)));
                    delete(fig);
            
                    ax = nexttile(2*j);
                    fig = copyobj(ax,figure);
                    set(fig,'OuterPosition',[0 0 1 1]);
                    xlswritefig(fig, filePath+"/"+"Modelled_Flow_Rates.xlsx", transformations{i}, strcat('A', int2str(testsToConduct+j),':','A', int2str(testsToConduct+j+5)));
                    delete(fig);
                end        
            end
        catch exception %#ok<NASGU>
            disp("Error: Excel not installed. Cannot write figure to xlsx")
        end
    end
    %-------- PERFORM TESTS ON CUMULATIVE FLOW RATES (just graphs) ---------%
    tiledlayout(ceil(length(transformations)/4),4)
    for i = 1:length(transformations)
        nexttile;
        create_flow_graph(squeeze(cumFlows(:,i,:)), stdFlows(:,i), treatments, alpha, "Plot of cumulative Flow "+ transformations{i});
    end
    %pause;    
    %-------- PERFORM TESTS ON POOL SIZES (just graphs) ---------%
    tiledlayout(ceil(length(poolNames)/4),4)
    for i = 1:length(poolNames)
        nexttile;
        create_pool_graph(squeeze(pools(:,i,1,:)), squeeze(stdPools(:,i,1,:)), squeeze(observations(:,i,1,:)), measuredTimes, treatments, poolNames{i}, alpha);
    end
    %pause;
    %--------- REPEAT TESTS WITH SIMULATED RESULTS ----- %
    %%Generate simulated samples
    [avgFlowSim, poolFlowSim, flowSim] = mcmc_sample(compareFile, ...
        transformations, ...
        transformationMatrix, ...
        100,...
        true);
    %-------- PERFORM TESTS ON AVEREAGE N RATES ---------%
    %generate bar chart for avgFlow (lower, upper, mean)
    tfig = tiledlayout(ceil(length(transformations)/4),4)
    for i = 1:length(transformations)
        nexttile;
        create_bar_Chart(avgFlowSim.mean(:,i), avgFlowSim.sd(:,i), treatments, transformations{i});
    end
    saveas(tfig, filePath+"/"+"Simulated_Avg_N_Rates","fig");
    %pause;
    %perform pairwise t-tests on simulated average n rates    
    avgNtest = struct;
    testsToConduct = nchoosek(length(treatments),2);
    allPvalues = zeros(testsToConduct * length(transformations),1);
    indxP = 1;
    for i=1:length(transformations)
        avgNtest.(transformations{i}) = create_results_table(treatments, avgFlowSim.mean(:,i), avgFlowSim.sd(:,i), nSamples, alpha);
        allPvalues(indxP:(i*testsToConduct)) = avgNtest.(transformations{i}).pValue;
        indxP = indxP + testsToConduct;
    end
    write_to_excel(avgNtest,tfig, filePath+"/"+"Simulated_Avg_N_Rates.xlsx");
    
    %-------- PERFORM TESTS ON FLOW RATES ---------%
    tfig = tiledlayout(ceil(length(transformations)/4),4)
    for i = 1:length(transformations)
        nexttile;
        create_flow_graph(squeeze(flowSim.mean(:,i,:)), squeeze(flowSim.sd(:,i,:)), treatments, alpha, "Plot of Simulated Flow Rate "+ transformations{i});
    end
    saveas(tfig, filePath+"/"+"Simulated_Flow_Rates","fig");    
    %pause;

    flowNTest = struct;
    tableColumnNames = cell(measuredTimes(end)+2,1);
    tableColumnNames{1} = 'Group1';
    tableColumnNames{2} = 'Group2';
    for i = 1:measuredTimes(end)
        tableColumnNames{2+i} = strcat('P(t=',int2str(i),')');
    end
    for i = 1:length(transformations)
        out = cell(testsToConduct,measuredTimes(end)+2);
        tfig = tiledlayout(ceil(testsToConduct / 2), 4)
        indx = 1;
        for j = 1:length(treatments)
            for k = (j+1):length(treatments)
                nexttile
                create_flow_graph(squeeze(flowSim.mean(:,i,[j k])), squeeze(flowSim.sd(:,i, [j k])), treatments([j k]), alpha, "Plot of Simulated Flow Rate "+ transformations{i}); 
                nexttile;
                [~, ~, ~, p] = independent_t_test(squeeze(flowSim.mean(:,i,j)), squeeze(flowSim.mean(:,i,k)), squeeze(flowSim.sd(:,i, j)), squeeze(flowSim.sd(:,i, k)), nSamples(j), nSamples(k), alpha);
                %perform pvalue adjustment
                p = pval_adjust(p, "none");
                yline(alpha, '-','Threshold')
                xlabel("Time");
                ylabel("P value");
                title(strcat("Probability of Equal Rate at Time T"));
                hold on
                scatter(1:length(p), p, "Marker", "x");
                hold off

                %Append to table
                out{indx, 1} = treatments{k};
                out{indx, 2} = treatments{j};
                for z = 2:(measuredTimes(end)+2)
                    out{indx, z} = p(z-1);
                end
                indx = indx + 1;
            end
        end
        %pause;
        saveas(tfig, filePath+"/"+"Simulated_Flow_Rate_"+transformations{i},"fig");
        flowNTest.(transformations{i}) = cell2table(out, 'VariableNames', tableColumnNames);
        writetable(flowNTest.(transformations{i}), filePath+"/"+"Simulated_Flow_Rates.xlsx", 'Sheet', transformations{i}, 'Range', 'A1');
    end
end

function [flows, cumFlows, stdFlows, ...
          avgFlows, pools, stdPools,...
          observations, poolNames, ...
          treatments, measuredTimes, ...
          alpha, pAdjustMethod,...
          transformations, transformationMatrix] = load_from_file(compareFile)
    
    [filepath, filename] = fileparts(compareFile);
    info = readtable(compareFile, 'Sheet', "Treatment_Info");
    config = readcell(compareFile, 'Sheet',"Configuration");
    
    alpha = config{strcmp('Significance Level',config(:,1)),2};
    pAdjustMethod = config{strcmp('P Value Adjustment',config(:,1)),2};

    parameterTable = readtable(compareFile, 'Sheet', 'Parameters');
    combinedParameterTable = readtable(compareFile, ...
                                        'Sheet', 'Combined Parameters');

    parameters = parameterTable.Transformation( ...
                            parameterTable{:,"PerformTest"} == 1);
    combinedParameters = combinedParameterTable.Combination( ...
                            combinedParameterTable{:,"PerformTest"} == 1);
    %We assume the same Ntrace model is used for all treatments
    load(filepath+"/"+info.Filename{1}, "modelFlows");

    transformationMatrix = zeros(length(parameters) + length(combinedParameters), ...
                                length(modelFlows.parameter));
    
    %Generate matrix that will extract/create transformations/lin combs
    indx = 1;
    for i = 1:length(parameters)
        transformationMatrix(indx,:) = strcmp(parameters{i}, ...
                                              modelFlows.parameter);
        indx = indx + 1;
    end
    for i = 1:length(combinedParameters)
        [positive, negative] = split_sum(combinedParameters{i});
        for j = 1:length(positive)
            transformationMatrix(indx,:) = transformationMatrix(indx,:) + ...
                                strcmp(positive{j}, modelFlows.parameter)';
        end
        for j = 1:length(negative)
            transformationMatrix(indx,:) = transformationMatrix(indx,:) - ... 
                                strcmp(negative{j}, modelFlows.parameter)';
        end
        indx = indx + 1;
    end
    %Get transformation names
    transformations = [
        cellfun(@(x)strrep(x,"-","_"), ...
                    parameterTable.Transformation( ...
                        parameterTable{:,"PerformTest"} == 1)); ...
         combinedParameterTable.Transformation( ...
                        combinedParameterTable{:,"PerformTest"} == 1)...
                        ];
    %Extract information from xlsx file    
    flows = [];
    cumFlows = [];
    stdFlows = [];
    avgFlows = [];
    pools = [];
    stdPools = [];
    observations = [];
    poolNames = [];
    measuredTimes = [];
    treatments = info.Treatment;
    for i=1:length(treatments)
        load(filepath+"/"+info.Filename{i}, "modelOutput", "measurements");
        if isempty(flows)
            flows = zeros(size(modelOutput.flowRate,1), length(transformations), length(treatments));
            cumFlows = zeros(size(modelOutput.flowRate,1), length(transformations), length(treatments));
            stdFlows = zeros(length(treatments), length(transformations));
            avgFlows = zeros(length(treatments), length(transformations));
            pools = zeros(size(modelOutput.modelOut,1), size(modelOutput.modelOut,2), size(modelOutput.modelOut,3), length(treatments));
            stdPools = zeros(size(modelOutput.modelOut,1), size(modelOutput.modelOut,2), size(modelOutput.modelOut,3), length(treatments));
            observations = zeros(size(measurements.obs,1), size(measurements.obs,2), size(measurements.obs,3), length(treatments));
            poolNames = measurements.measuredVariableNames;
            measuredTimes = measurements.time;
        end
        for j = 1:length(transformations)
            flows(:,j,i) = sum(transformationMatrix(j,:) .* modelOutput.flowRate, 2);
            cumFlows(:,j,i) = sum(transformationMatrix(j,:) .* modelOutput.cumFlow, 2);
            stdFlows(i,j) = sum(transformationMatrix(j,:) .*  fillmissing(modelOutput.stdRates(:).', 'constant',0));
            avgFlows(i,j) = sum(transformationMatrix(j,:) .* fillmissing(modelOutput.avgFlowRate(:).', 'constant',0));
        end
            pools(:,:,:,i) = modelOutput.modelOutPools;
            stdPools(:,:,:,i) = modelOutput.stdPools;
            observations(:,:,:,i) = measurements.obs;
    end
end

function [positive, negative] = split_sum(str)
    % Split linear combination into positive and negative values
    % Input: str string String to split
    % Output: positive array Positive part of split string
    %         negative array Negative part of split string
    % Description: Split linear combination string 'str' into two arrays.
    % One contains the positive values and the other negative values.
    % Example:
    %   str = 'a + b - c - d';
    %   [positive, negative] = split_sum(str);
    %   positive: [a,b]
    %   negative: [c,d]
    % Note: str must have spaces between + or - (e.g. 'a + b' not 'a+b').
    
    positive = split(str,' + ');
    [negative, m] = split(positive(end), ' - ');
    if ~isempty(m)
        positive(end) = negative(1);
        negative = negative(2:end);
    else
        negative = {};
    end
end

function create_bar_Chart(rates, stdrates, names, flowName)
%Create barchart to visually compare avg N rates
    bar(names, rates);
    hold on
    errorbar(rates, stdrates, '.');
    title(strcat("Average N rate: ",flowName));
    xlabel("Treatments");
    ylabel("Avg N Rate");
    hold off
end

function create_flow_graph(flowRates, stdRate, names, alpha, plotTitle)
%Create plot of flow rates to visually compare flows
    zstat = norminv(1-alpha/2);
    hold on 
    xconf = horzcat((1:1:size(flowRates,1)), (size(flowRates,1):-1:1));         
    for i = 1:length(names)
        plot(flowRates(:,i), 'Color', [randi(5)/5 1-i/5 i/5]);

        yconf = [flowRates(:,i).' + zstat*stdRate(i) flowRates(end:-1:1,i).' - zstat*stdRate(i)];
        fill(xconf, yconf, [randi(5)/5 1-i/5 i/5], 'EdgeColor', 'none', 'FaceAlpha', 0.25, 'HandleVisibility','off')       

    end
    xlabel("Time");
    ylabel("N Transformation Rate");
    legend(names);
    title(plotTitle)
    hold off
end

function create_pool_graph(poolFlows, stdPools, measurements, times, names, poolName, alpha)
%Create plot of pools with measurements to visually compare pool flows
    mrk = {'d','x','*','+'};
    zstat = norminv(1-alpha/2);
    hold on
    xconf = horzcat((1:1:size(poolFlows,1)), (size(poolFlows,1):-1:1));         
    for i = 1:length(names)
        plot(poolFlows(:,i), 'Color',[randi(5)/5 1-i/5 i/5]);
        yconf = [poolFlows(:,i).' + zstat*stdPools(:,i).' poolFlows(end:-1:1,i).' - zstat*stdPools(end:-1:1,i).'];

        fill(xconf, yconf, [randi(5)/5 1-i/5 i/5], 'EdgeColor', 'none', 'FaceAlpha', 0.25, 'HandleVisibility','off');

    end
    legend(names);
    for i = 1:length(names)
        scatter(times, measurements(:,i), 'Color',[randi(5)/5 1-i/5 i/5], ...
                'Marker',mrk{i},'HandleVisibility','off');
    end
    xlabel("Time");
    ylabel("N Pool Size");
    title(strcat("Pools and measurements: ", poolName));
    hold off    
end

function [difference, lower, upper, p] = independent_t_test(mu1, mu2, sd1, sd2, n1, n2, alpha)
    %Perform indepedent 2-sample t-tests
    df = n1+n2-2;
    tcrit = tinv(1-alpha/2, df);
    sPooled = sqrt( ((n1-1)*sd1.^2 + (n2-1)*sd2.^2)/((n1+n2-2)*(1/n1 + 1/n2)) );
    difference = mu1 - mu2;
    T = abs(difference)./sPooled;
    lower = difference - tcrit*sPooled;
    upper = difference + tcrit*sPooled;
    p = tcdf(T,df, 'upper');
end

function write_to_excel(tables, tiledFigure, fileName)
    %Write table of results to excel file
    %tables is a struct of tables
    transformationNames = fieldnames(tables);
    for i = 1:length(transformationNames)
        %Extract table with test results
        tbl = tables.(transformationNames{i});
        writetable(tbl,fileName, 'Sheet',transformationNames{i}, 'Range', 'A1');
        %Extract figure from tiled layout and to excel sheet
        if ishandle(tiledFigure)
            fig = figure();
            ax = nexttile(i);
            copyobj(ax,fig);
            set(fig,'OuterPosition',[0 0 1 1]);
            %xlswritefig(fig, fileName, transformationNames{i}, 'A9');
            delete(fig);
        end
    end
end

function tbl = create_results_table(tNames, mu, std, n, alpha)
    %perform pairwise t-tests and form table of results
    indx = 1;
    out = cell(nchoosek(length(tNames), 2), 6);
    for j = 1:length(tNames)
        for k = (j+1):length(tNames)
            [difference, lower, upper, p] = independent_t_test(mu(j), mu(k), std(j), std(k), n(j), n(k), alpha);
            out{indx, 1} = tNames(j);
            out{indx, 2} = tNames(k);
            out{indx, 3} = lower;
            out{indx, 4} = difference;
            out{indx, 5} = upper;
            out{indx, 6} = p;
            indx = indx + 1;
        end
    end
    tbl = cell2table(out, ...
        'VariableNames', ...
        {'Group1','Group2','LowerLimit_CI','GroupDifference','UpperLimitCI','pValue'});        
end

function pc = pval_adjust(p, method)
    % PVAL_ADJUST Adjust p-values for multiple comparisons. Given a set of
    % p-values, returns p-values adjusted using one of several methods.    %
    % Copyright (c) 2016 Nuno Fachada
    % Distributed under the MIT License (See accompanying file LICENSE or copy 
    % at http://opensource.org/licenses/MIT)

    % Number of p-values
    np = numel(p);
    % Reshape input into a row vector, keeping original shape for later
    % converting results into original shape
    pdims = size(p);
    p = reshape(p, 1, np);
    % Method 'hommel' is equivalent to 'hochberg' of np == 2
    if (np == 2) &&  strcmp(method, 'hommel')
        method = 'hochberg';
    end
    % Just one p-value? Return it as given.
    if np <= 1
        pc = p;
    % What method to use?
    elseif strcmp(method, 'holm')
        % Sort p-values from smallest to largest
        [pc, pidx] = sort(p);
        [~, ipidx] = sort(pidx);        
        % Adjust p-values
        pc = min(1, cummax((np - (1:np) + 1) .* pc));
        % Put p-values back in original positions
        pc = pc(ipidx);
    elseif strcmp(method, 'hochberg')
        % Descendent vector
        vdec = np:-1:1;
        % Sort p-values in descending order
        [pc, pidx] = sort(p, 'descend');
        % Get indexes of p-value indexes
        [~, ipidx] = sort(pidx);
        % Hochberg-specific transformation
        pc = ((np + 1) - vdec) .* pc;
        % Cumulative minimum
        pc = cummin(pc);
        % Reorder p-values to original order
        pc = pc(ipidx);
    elseif strcmp(method, 'hommel')
        % Sort p-values from smallest to largest
        [pc, pidx] = sort(p);        
        % Get indexes of p-value indexes
        [~, ipidx] = sort(pidx);
        % Generate vectors for cycle
        pa = repmat(min(np * pc ./ (1:np)), size(p));
        q = pa;        
        % Begin cycle
        for i = (np - 1):-1:2
            i1 = 1:(np - i + 1);
            i2 = (np - i + 2):np;
            q1 = min(i * pc(i2) ./ (2:i));
            q(i1) = min(i * pc(i1), q1);
            q(i2) = q(np - i + 1);
            pa = max(pa, q);
        end        
        % Finalize result
        pa = max(pa, pc);
        pc = pa(ipidx);
    elseif strcmp(method, 'bonferroni')        
        % Simple conservative Bonferroni
        pc = p * numel(p);
    elseif strcmp(method, 'BH') || strcmp(method, 'fdr')
        % Descendent vector
        vdec = np:-1:1;        
        % Sort p-values in descending order
        [pc, pidx] = sort(p, 'descend');
        % Get indexes of p-value indexes
        [~, ipidx] = sort(pidx);
        % BH-specific transformation
        pc = (np ./ vdec) .* pc;
        % Cumulative minimum
        pc = cummin(pc);            
        % Reorder p-values to original order
        pc = pc(ipidx);
    elseif strcmp(method, 'BY')
        % Descendent vector
        vdec = np:-1:1;        
        % Sort p-values in descending order
        [pc, pidx] = sort(p, 'descend');
        % Get indexes of p-value indexes
        [~, ipidx] = sort(pidx);
        % BY-specific transformation
        q = sum(1 ./ (1:np));
        pc = (q * np ./ vdec) .* pc;
        % Cumulative minimum
        pc = cummin(pc);            
        % Reorder p-values to original order
        pc = pc(ipidx);
    elseif strcmp(method, 'sidak')
        % Sidak correction
        pc = 1 - (1 - p) .^ np;        
    elseif strcmp(method, 'none')        
        % No correction
        pc = p;        
    else        
        % Unknown method
        disp('Unknown p-value adjustment method. Default to None.');        
        pc = p;        
    end
    % Can't have p-values larger than one
    pc(pc > 1) = 1;    
    % Reshape result vector to original form
    pc = reshape(pc, pdims);

end