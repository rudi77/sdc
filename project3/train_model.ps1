$data_base = '.\data_base\driving_log.csv'
$data_curves ='.\data_curves\driving_log.csv'
$data_recovery = '.\data_recovery\driving_log.csv'
$data_smooth_jungle = '.\data_smooth_jungle\driving_log.csv'
$data_smooth_lake = '.\data_smooth_lake\driving_log.csv'
$data_straightahead = '.\data_straightahead\driving_log.csv'

[string]$logfiles = "$data_base,$data_curves,$data_recovery,$data_smooth_jungle,$data_smooth_lake,$data_straightahead"

python.exe ./model.py -i $logfiles -e 10