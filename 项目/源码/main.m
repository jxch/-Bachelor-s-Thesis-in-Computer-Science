function [] = main(func)
    switch(func)
        case 'NPE'
            run('./mainNPE.m');
        case 'ONPE'
            run('./mainONPE.m');
        case 'IsoP'
            run('./mainIsoP.m');
        case 'OIsoP'
            run('./mainOIsoP.m');
        case 'LSDA'
            run('./mainLSDA.m');
        case 'OLSDA'
            run('./mainOLSDA.m');
        otherwise
            error('Êý¾Ý´íÎó');
    end
end

