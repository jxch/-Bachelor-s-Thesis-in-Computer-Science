      fea = rand(50,70);
      gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
      options = [];
      options.k = 5;
      options.NeighborMode = 'Supervised';
      options.gnd = gnd;
      [eigvector, eigvalue] = IsoP(options, fea);