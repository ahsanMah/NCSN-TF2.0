for i in 1 3 5 7 9
do
    CUDA_VISIBLE_DEVICES=0 python runaux.py 1 $i &
    CUDA_VISIBLE_DEVICES=1 python runaux.py 1 $( expr $i + 1 )
done
