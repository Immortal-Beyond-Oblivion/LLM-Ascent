| Metric | character | bpe |
|---|---:|---:|
| Vocabulary size | 66 | 166 |
| BPE merges | 0 | 100 |
| Parameters | 108,416 | 114,816 |
| Characters per token (validation) | 1.000 | 1.620 |
| Train loss (per token) | 2.3961 | 3.4269 |
| Validation loss (per token) | 2.3931 | 3.4757 |
| Validation perplexity (per token, NOT cross-comparable) | 10.95 | 32.32 |
| Validation bits per character | 3.4525 | 3.0946 |
| Validation perplexity (per character) | 10.947 | 8.542 |
| Training wall-clock (s) | 5.0 | 28.2 |
| Greedy repetition (char 3-gram) | 0.970 | 0.742 |
| Sampled repetition (char 3-gram) | 0.126 | 0.091 |
| Greedy tokens/s | 3429.0 | 3119.0 |
| Greedy characters/s | 3429.0 | 4148.2 |
| Sampled characters/s | 3144.4 | 4277.2 |

**character / greedy**
```
ROMEO:
The the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

**character / top-k sampling (k=20, T=0.8)**
```
ROMEO:
I maknd ing s cheaceind ak mourd thy heal y, whell hathe thigre hed t
Whin on swhas sthe llr thatofange rimy murentr h:
Thisis be te is thalle ranothat mir our go mmuspicesss it ind,
I cond s thabeou
```

**bpe / greedy**
```
ROMEO: I:
Whe with come with cell my should thould thould thould thould the
To should thould thould thould thould thou cema cema cema cema cema cema ce
To with cema cema cem cem cem cem cem cececem cem cem 
```

**bpe / top-k sampling (k=20, T=0.8)**
```
ROMEO: fill com to so ain,
Sple doly bedon the gaf I amande
colfrownecexten her and geventol, wid be thencarting forcriswily ce se sso withould siondys fend,
And spar of gartim till k sut's cker, to t
Thet 
```
