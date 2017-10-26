function [ pp ] = configure_parpool( numWorkers )
%CONFIGURE_PARPOOL Creates a new parpool with specified workers
%(numWorkers). If a pool already exists, the pool is maintained.
%If numWorkers is not supplied, the parpool is initiated using the default
%parpool.

if isempty(gcp('nocreate'))
    if exist('numWorkers')
        warning(strcat('Check worker count, currently: ', num2str(numWorkers)))
        pp = parpool(numWorkers);
    else
        pp = parpool;
    end
else
    pp = gcp;
end

end

