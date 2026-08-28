# sfi_fno_bioacoustics
Mad Scientist tests on fancy bioacoustic networks

### Modulated siren training:
```bash
source venv/bin/activate && python3 train.py --model-type film_siren --num-epochs 100 --batch-size 32 --device cuda --optimizer adamw --learning-rate 1e-3 --weight-decay 1e-4
```
