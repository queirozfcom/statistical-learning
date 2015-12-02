function gridworld()

  gamma = 0.9;
  theta = 10.0^-5;
  observations = 0;
  
  function observe(s, a)
    s = s - 1;
    y = floor(Int64, s/5);
    x = s-y*5;
    
    observations = observations + 1;

    if y == 0
      if x == 1
        return (10, 4*5+1+1);
      elseif x == 3
        return (5, 2*5+3+1);
      end
    end

    if a == 1
      y = y - 1;
    elseif a == 2
      x = x + 1;
    elseif a == 3
      y = y + 1;
    elseif a == 4
      x = x - 1;
    else
      error("Unknown action");
    end

    r = 0;

    if x < 0
        r = -1;
        x = 0;
    elseif x > 4
        r = -1;
        x = 4;
    elseif y < 0
        r = -1;
        y = 0;
    elseif y > 4
        r = -1;
        y = 4;
    end

    sp = y*5+x+1;
    
    return (r, sp);
  end

  function evaluate(pi)
    # TODO: Implement policy evalation
    return v;
  end

  function improve(v)
    # TODO: Implement policy improvement
    return pi;
  end

  lastpi = zeros(1, 25);
  pi = lastpi + 1;

  v = lastpi;
  while any(lastpi != pi)
    lastpi = copy(pi);
    v = evaluate(pi);
    pi = improve(v);
  end
  
  return (reshape(v, 5, 5)', observations);
    
end