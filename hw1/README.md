# HW1
Roop Pal (rmp2191)

## Question 4
### Running
```
python count_freqs.py ner_train.dat > ner.counts

python q4p12.py

python count_freqs.py modified_ner_train.dat > modified_ner.counts

python q4p3.py

python eval_ne_tagger.py ner_dev.key 4.txt
```
### Performance
```
Found 14043 NEs. Expected 5931 NEs; Correct: 3117.
         precision      recall          F1-Score
Total:   0.221961       0.525544        0.312106
PER:     0.435451       0.231230        0.302061
ORG:     0.475936       0.399103        0.434146
LOC:     0.147750       0.870229        0.252612
MISC:    0.491689       0.610206        0.544574
```

Unsurprisingly such a naive method has very low overall f-1 scores.
It's interesting to note that recall outperforms precision.

## Question 5
### Running
```
python q5p1.py

python q5p2.py

python eval_ne_tagger.py ner_dev.key 5_2.txt
```

### Performance

```

PERFORMANCE
Found 4661 NEs. Expected 5931 NEs; Correct: 3657.
         precision      recall          F1-Score
Total:   0.784596       0.616591        0.690521
PER:     0.744311       0.605005        0.667467
ORG:     0.659729       0.473842        0.551544
LOC:     0.887883       0.695202        0.779817
MISC:    0.825974       0.690554        0.752218
```

The performance is not terrible, with higher precision than recall.

## Question 6
### Classes chosen

- `_DIGIT_` if all characters are numerals
- `_PARTNUM_` if any single character is a numeral
- `_DASHES_` if all characters are dashes
- `_ABBREV_` if it is an abbreviation (i.e. "NY", "U.N.")
- `_PROPER_` if only the first letter is capitalized
- `_RARE_` for all other cases


### Running
```
python q6p1.py

python count_freqs.py q6_ner_train.dat > q6_ner.counts

python q6p2.py

python eval_ne_tagger.py ner_dev.key 6.txt
```
### Performance

```
Found 5596 NEs. Expected 5931 NEs; Correct: 4292.

         precision      recall          F1-Score
Total:   0.766976       0.723655        0.744686
PER:     0.820469       0.780740        0.800112
ORG:     0.579114       0.683857        0.627142
LOC:     0.869886       0.707197        0.780150
MISC:    0.831186       0.700326        0.760165
```


Though the precision decreased, the recall greatly increased, giving the best f1-scores.