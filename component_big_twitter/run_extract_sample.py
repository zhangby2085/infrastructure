# run_extract_sample.py

import sframe
sf_full = sframe.SFrame.read_csv('data/01-raw/twitter.csv', header=False)
print sf_full.head()

for power in range(2,5):
  # print 
  sf_sample = sf_full.sample(pow(.1,power), seed=5)
  sf_sample.save('data/01-raw/sample-%s.csv' % pow(.1,power), format='csv')
