#!/bin/bash
cd plotdata
file="tmp"
file2="tmp2"
for i in $( seq -w 1 1 250 ); do
#    touch $i
#    for j in $( seq 0 3 ); do 
#        cat "Output_P_"$j"_t_"$i >> "step$i"  # filename example: Output_P_1_t_50
#    done

    touch $file
#    echo "set pm3d" >$file
#    echo "set dgrid3d 100,100,2" >> $file
#    echo "set hidden3d" >> $file
#    echo "set palette defined" >> $file
#    if [ $i -lt "6" ]; then
#        echo "less than six!"
#        echo "set cbrange[0:1]" >> $file
#    else
#        echo "more than six!"
#        echo "set cbrange[0:0.01]" >> $file
#    fi
#    echo "set xrange[0:1]" >> $file
#    echo "set yrange[0:1]" >> $file
#    echo "set zrange[
    echo "set terminal png"         >> $file
    echo "set output 'bilde$i.png'"   >> $file
    echo "splot \\" >> $file
    for j in 0 1 2 ; do
        echo  "'p"$j"_field"$i"' every 16,\\" >> $file
    done
    echo  "'p3_field"$i"' every 16" >> $file

#    echo "splot '"$file2"' every 3" >> $file 
    gnuplot $file

    rm $file
#    rm $file2
#    rm "step$i"
    echo  "step $i done"
done
mencoder mf://*.png -mf w=800:h=600:fps=5:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o ../field.avi
cd ..
