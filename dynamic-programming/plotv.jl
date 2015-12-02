function plotv(v)

    function snext(s, a)
        s = s - 1;
        y = floor(Int64, s/5);
        x = s-y*5;

        if y == 0
            if x == 1
                return 4*5+1+1;
            elseif x == 3
                return 2*5+3+1;
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

        if x < 0
            x = 0;
        elseif x > 4
            x = 4;
        elseif y < 0
            y = 0;
        elseif y > 4
            y = 4;
        end

        return y*5+x+1;
    end

    v = reshape(v', 1, 25);
    pi = zeros(1, 25);
    for s = 1:25
        actions = zeros(1, 4);
        for a = 1:4
            sp = snext(s, a);
            actions[a] = v[sp];
        end
        pi[s] = indmax(actions);
    end

    v = reshape(v, 5, 5)';
    pi = reshape(pi, 5, 5)';
    
    ax = -sign(pi-3).*rem(pi-1, 2);
    ay = sign(pi-2).*rem(pi, 2);
    
    p = quiver(collect(1:5), collect(1:5), ax, ay, 0.5);
    hold(true)
    
    for xx=1:5
        for yy=1:5
            text(xx, yy, "$(@sprintf "%5.1f" v[yy,xx])", horizontalalignment="center");
        end
    end
    
    hold(false)
    axis([0, 6, 0, 6]);

    title("Gridworld V(s) and \pi(s)");
    xlabel("Column");
    ylabel("Row");
    
end
    