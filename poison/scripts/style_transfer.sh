set -e
args=$(getopt -o "" -l sample_dir:,aim_style:,model_dir:,model_type: -- "$@")
args=$(echo "$args" | tr -d "'")
set -- $args
while true;do
    case "$1" in
        --sample_dir)
            sample_dir="$2"
            shift 2 ;;
        --aim_style)
            aim_style="$2"
            shift 2 ;;
        --model_dir)
            model_dir="$2"
            shift 2 ;;
        --model_type)
            model_type="$2"
            shift 2 ;;    
        --)
            break ;;
        *)
            echo "invalid params or options"
            exit 1
    esac
done

for style_dir in ${sample_dir}/*; do
    for artist_dir in ${style_dir}/*; do
        python ../style_transfer.py --aim_style=${aim_style} --artist_dir=${artist_dir} \
                                    --model_dir=${model_dir} --model_type=${model_type}
    done
done
