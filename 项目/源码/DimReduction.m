function [eigvector, eigvalue] = DimReduction(options, allsamples)
    switch(options.func)
        case 'NPE'
            [eigvector, eigvalue] = NPE(options, allsamples);
        case 'ONPE'
            [eigvector, eigvalue] = ONPE(options, allsamples);
        case 'IsoP'
            [eigvector, eigvalue] = IsoP(options, allsamples);
        case 'OIsoP'
            [eigvector, eigvalue] = OIsoP(options, allsamples);
        case 'LSDA'
            [eigvector, eigvalue] = LSDA(options, allsamples);
        case 'OLSDA'
            [eigvector, eigvalue] = OLSDA(options, allsamples);
        otherwise
            error('Êý¾Ý´íÎó');
    end
end

