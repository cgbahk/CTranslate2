## ct2-inspect

Let's inspect how CTranslate2 translate optionsa affect performance
- How memory footprint determined?
    - Conjecture: `[memory footprint] ~ [sentence count in a batch] x [max sentence length]**2`
