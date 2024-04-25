data_dir='../data'
results_dir='../results'
rm -rf data
rm -rf results
ln -s ${data_dir} data
if [ ! -d "${results_dir}/luke/results" ]; then
    mkdir -p ${results_dir}/luke/results
fi
ln -s ${results_dir}/luke/results results
