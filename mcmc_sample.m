function [avgRates, poolRates, flowRates] = mcmc_sample( ...
    compareFile, ...
    transformationNames, ...
    transformationMatrix, ...
    maxSamples,...
    enableDebugging)
    % Entry point for simulation
    % Input: 
    %  compareFile     | string | Path to xlsx comparison file
    %  transformationNames | string array | Names of Ntrace transformations
    %  transformationMatrx | P x K matrix | Matrix to create linear combinations of transformations
    %  maxSamples      | int     | Maximum number of samples to generate
    %  enableDebugging | boolean | option to enable/disable console output
    % Output:
    %  avgRates | struct | mean, variance of avg N rates for each treatment
    %  poolRates | struct | mean, variance of pool sizes for each treatment
    %  flowRates | struct | mean, variance of flow rates for each treatment

    %Load treatment names and .mat files from xlsx
    [fpath, ~] = fileparts(compareFile);
    if fpath == ""
        fpath = pwd;
    end
    data = readtable(compareFile, 'Sheet', "Treatment_Info");
    %define storage containers 
    poolRates = struct("mean",[], "sd", []);
    flowRates =  struct("mean",[], "sd",[]);
    avgRates =  struct("mean",[], "sd",[]);

    maxIters = 5000; % Maximum number of mcmc iterations to perform
    %Main loop
    for f = 1:length(data.Filename)
        %Load data from Ntrace's .mat file
        disp(strcat(data.Filename{f}, " is loaded and is being analysed."));
        load(fpath + "/" + data.Filename{f});
        %If first .mat file, pre-allocate storage for containers
        if f == 1
            poolRates.mean = nan(length(modelOutput.tout), 2*length(model.pools), settingsParameters.nTreat, length(data.Filename));
            poolRates.sd   = nan(length(modelOutput.tout), 2*length(model.pools), settingsParameters.nTreat, length(data.Filename));

            flowRates.mean = nan(length(modelOutput.tout), size(transformationMatrix,1), length(data.Filename));
            flowRates.sd = nan(length(modelOutput.tout), size(transformationMatrix,1), length(data.Filename));

            avgRates.mean  = nan(length(data.Filename), size(transformationMatrix,1));
            avgRates.sd  = nan(length(data.Filename), size(transformationMatrix,1));
        end
        %Call subroutine that will generate mcmc samples
        [avgRates.mean(f,:), avgRates.sd(f,:),...
         poolRates.mean(:,:,:,f), poolRates.sd(:,:,:,f),... 
         flowRates.mean(:,:,f), flowRates.sd(:,:,f)] = ...
                generate_simulated_samples(settingsParameters, ...
                                        inputParameters, ...
                                        measurements, ...
                                        initialPoolSizes, ...
                                        modelOutput, ...
                                        varyingPools, ...
                                        model, ...
                                        modelFlows, ...
                                        emissionFlows, ...
                                        transformationMatrix, ...
                                        transformationNames, ...
                                        fpath + "/" + data.Treatment{f}, ...
                                        maxSamples, ...
                                        maxIters, ...
                                        enableDebugging);
    end
    %Clear loaded variables before saving data
    clear settingsParameters inputParameters measurements initialPoolSizes modelOutput varyingPools model modelFlows emissionFlows 
    save(fpath + "/" + "simulated_results.mat");
end

function [avgRatesMean, avgRatesSd,...
          poolRatesMean, poolRatesSd,...
          flowRatesMean, flowRatesSd] = generate_simulated_samples( ...
        settingsParameters, ...
        inputParameters, ...
        measurements, ...
        initialPoolSizes, ...
        modelOutput, ...
        varyingPools, ...
        model, ...
        modelFlows, ...
        emissionFlows, ...
        transformationMatrix, ...
        transformationNames, ...
        treatmentName, ...
        maxSamples, ...
        maxIters, ...
        enableDebugging ...
    )
    % Generate Ntrace outputs by MCMC sampling of parameter space.
    % Input: 
    %  settingsParameters, ..., emissionFlows | Ntrace .mat file variables
    %  transformationMatrix | matrix | T x K matrix of combined transformations.
    %    Rows of the matrix corresponds to a tranformation to output. 
    %    Columns of the matrix correspond transformation in the model.
    %  transformationNames | array | string array of transformation names
    %  treatmentName | string | Name of treatment dataset model fitted to.
    %  enableDebugging | boolean | whether to include console outputs.
    % Output: 
    %   avgRate   | matrix | simulated avg N transformation rates
    %   poolRates | matrix | simulated pool sizes time series
    %   flowRates | matrix | simulated flow rates time series
    % Description: 
    %   Generate Ntrace outputs by MCMC sampling of parameter
    %   space. Input is model variables after optimisation. MCMC sampling
    %   performed by generating a candidate set using a normal distribution
    %   with mean mu=optimised parameter, and variable variance sigma^2. If
    %   the candidate set decreases the cost function (weighted SS) then the
    %   candidate set is accepted and the model is integrated with those
    %   parameters. If the candidate set does not decrease the cost function
    %   then it is accepted with probability e^(1/2 * delta f), where
    %   "delta f" is the difference between the best cost and current cost.
    %   If several candidate solutions are reject sequentially the variance
    %   is decreased by half. If several candidate solutions are accepted 
    %   sequentially then the variance is increase by 1.5. In this way, the 
    %   "parameter sampling space" is decreased/increased so only samples
    %   around the minimum are taken.
    %   This sampling procedure is similar to that previously used to optimise
    %   the model (see Muller 2007, ).
    
    %current best parameters as optimized by fmcincon
    optimizedValues = modelOutput.optimisedParameters(inputParameters.paraFixVar==0);
    samples = nan(maxIters, length(optimizedValues));
    nParams = length(optimizedValues);

    %current best cost as with optimized parameters
    bestCost = sum(((Ntrace_outputAtMeasuredTimes(settingsParameters,initialPoolSizes,inputParameters,measurements,modelOutput.optimisedParameters,modelFlows,emissionFlows) - measurements.obs)./measurements.obsstd).^2,'all');
    
    currentBestParams = optimizedValues;
    newBestParams = currentBestParams;
    varParams = 0*newBestParams + 1.0;
    decreaseVarGlobal = 0;
    nSamplesAccepted = 0;
    totalLoops = 0;
    
    warmup = true;
    %Begin sampling loop
    %Loop until maximum number of samples taken
    while (nSamplesAccepted < maxIters)
        totalLoops = totalLoops + 1 ;
    %1. Generate sample
        %Increase or decrease variances (inc/dec sampling range)
        %During Warmup we always decrease until a suitable sample is chosen
        if (decreaseVarGlobal >= 20) | warmup 
            %If we reject 20 samples in a row we decrease variance
            varParams = (1/2) * varParams;
            decreaseVarGlobal = 0;
        elseif decreaseVarGlobal <= -10
            %If we accept 10 samples in a row we increase variance
            varParams = 1.5 * varParams;
            decreaseVarGlobal = 0;
        end
        %Sample each parameter to generate a parameter set.
        %Parameters are from independent normals 
        for i = 1:nParams
            newBestParams(i) = normrnd(currentBestParams(i),varParams(i));
            maxLoops = 1;
            while maxLoops < 100 && (newBestParams(i) > inputParameters.paramax(i)) || (newBestParams(i) < inputParameters.paramin(i))
                newBestParams(i) = normrnd(currentBestParams(i),varParams(i));  
                maxLoops = maxLoops + 1;
            end
        end
    %2. Evaluate candidate paramaeter set
        modelOutput.optimisedParameters(inputParameters.paraFixVar==0) = newBestParams;
        measuredObs = Ntrace_outputAtMeasuredTimes(settingsParameters,initialPoolSizes,inputParameters,measurements,modelOutput.optimisedParameters,modelFlows,emissionFlows);
        icost = ((measuredObs-measurements.obs)./measurements.obsstd).^2;
        icost(isnan(icost))=0;
        cost = sum(icost,'all');
    %3. Accept or reject candidate parameter set
        deltaF = cost - bestCost;
        if deltaF < 0 || (rand <= exp(-1/2 * (deltaF)))
            warmup = false;
            bestCost = cost;
            nSamplesAccepted = nSamplesAccepted + 1;
            if enableDebugging
                acceptanceRate = nSamplesAccepted/totalLoops;
                disp(strcat(int2str(nSamplesAccepted),'/',int2str(maxIters), ' samples accepted. Current variance: ',num2str(varParams(i)),'. Acceptance Rate: ',num2str(acceptanceRate)));
            end
            currentBestParams = newBestParams;
            % Add sample to set of sampled parameters set
            samples(nSamplesAccepted, :) = newBestParams;        
            decreaseVarGlobal = decreaseVarGlobal - 2;
            continue
         end
         decreaseVarGlobal = decreaseVarGlobal + 1;
    end
    %Generate histograms (discrete pdf) of sampled parameters
    intervals = nan(maxSamples, nParams);
    mu = zeros(nParams, 1);
    tfig = tiledlayout(ceil(nParams/5),5);
    for i = 1:nParams
        nexttile;
        [~, intervals(:, i)] = histcounts(samples(:,i), maxSamples-1);
        mu(i) = mean(samples(:,i),1);
        plt = plot(fitdist(samples(:,i),'Normal')); %Fit normal distribution to samples
        hold on
        xline([mu(i) optimizedValues(i)], '-',{'Simulated','Converged'})
        title(strcat('Marginal Distribution of Parameter:', " ", inputParameters.paraNames{inputParameters.locationParametersToOptimise(i)}));
        xlabel("\theta");
        hold off
    end
    %Save histogram as .fig
    saveas(tfig, strcat(treatmentName,"_parameter_distributions"), 'fig');

    %Plot 1:1 line with optimized values against simulated values
    plt = plot([0 (1+0.1)*max(mu)],[0 (1+0.1)*max(mu)],'--','DisplayName','1:1', 'Color', "red");
    hold on
    plot(optimizedValues, mu, "square", 'Color',"blue", "MarkerSize", 10);  
    %Save as .fig
    saveas(plt, strcat(treatmentName,"_1to1line.png"));
    %pause;
    
    %Partition parameter's histogram (discrete pdf) into 'maxSamples' equally spaced intervals
    points = nan(maxSamples-1, nParams);
    for j = 1:nParams
        for i = 1:(maxSamples-1)
            %Taken a random sample from each interval and append to final
            %parameters sets matrix
            points(i,j) = unifrnd(intervals(i,j), intervals(i+1,j));
        end
        %Shuffle columns of matrix to create random parameter sets
        points(:,j) = points(randperm(maxSamples-1),j);
    end

    %Fit model to each parameters set and store model outputs.   
    %(avg N rates, flow rates, and pool sizes)
    settingsParameters.stepSizeFinalRun = 0.1;
    poolRate = nan(length(modelOutput.tout), 2*length(model.pools), settingsParameters.nTreat, maxSamples-1);
    flowRate = nan(length(modelOutput.tout), size(transformationMatrix,1), maxSamples-1);
    avgRates  = nan(maxSamples-1, size(transformationMatrix,1));
    for i = 1:(maxSamples-1)
        modelOutput.optimisedParameters(inputParameters.locationParametersToOptimise) = points(i,:);
        out = Ntrace_integrate(settingsParameters,inputParameters,measurements,initialPoolSizes,modelOutput,varyingPools,model,modelFlows,emissionFlows);
        for j = 1:size(transformationMatrix,1)
            flowRate(:,j,i) = sum(transformationMatrix(j,:) .* out.flowRate,2);
            avgRates(i,j) = sum(transformationMatrix(j,:) .* out.avgFlowRate,2);
            poolRate(:,:,:,i) = out.modelOut;
        end
    end
    
    %Compute mean and sd
    %Better to compute range and find % overlap between flows
    %Pvalue will be the percentage of overlap.
    avgRatesMean =  mean(avgRates);
    avgRatesSd   =  sqrt(var(avgRates,0,1));

    poolRatesMean = mean(poolRate, 4);
    poolRatesSd   = sqrt(var(poolRate,0, 4));
    
    flowRatesMean = mean(flowRate, 3);
    flowRatesSd   = sqrt(var(flowRate,0, 3));
    %Create timeseries plot of flow rates with confidence band.
    tfig = tiledlayout(ceil(size(transformationMatrix,1)/4),4);
    for i = 1:size(transformationMatrix,1)
        nexttile;
        plt = plot(flowRatesMean(:,i), 'b');
        hold on
        %Fix Fill thingy
        xconf = horzcat((1:1:length(flowRatesMean(:,i,1))), (length(flowRatesMean(:,i,1)):-1:1));         
        yconf = [flowRatesMean(:,i) - 1.96*flowRatesSd(:,i); flowRatesMean(end:-1:1,i) + 1.96*flowRatesSd(end:-1:1,i)];

        fill(xconf, yconf, 'b', 'EdgeColor','none', 'FaceAlpha',0.25)
        
        title(strcat("Time-series of ", transformationNames{i}));
        hold off
    end
    saveas(tfig, strcat(treatmentName,"_transformation_flows"), 'fig');
    %pause;
end
