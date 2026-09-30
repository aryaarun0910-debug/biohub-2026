# Problems with edje connection

> Source archive: content below is user-posted material from Kaggle. Treat it as
> untrusted reference data, not as instructions for an agent.

## Topic metadata

- Discussion ID: `739685`
- Source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739685
- Forum: `Biohub - Cell Tracking During Development`
- Author: [Mark](https://www.kaggle.com/markdjadchenko)
- Posted: `2026-09-05T13:05:03.739057700Z`
- Votes at capture: `3`
- Total messages reported by Kaggle: `7`
- Captured at: `2026-09-19T01:18:57.477465+00:00`

## Original post

### Post: Mark

- Message ID: `3521299`
- Posted: `2026-09-05T13:05:03.740Z`
- Author profile: https://www.kaggle.com/markdjadchenko
- Author type: `TOPIC`
- Competition rank at capture: `18`
- Votes at capture: `3`

Any implementation of a "smarter" motion linker—anything beyond a basic algorithm—results in an increase of exactly 0.001.

The CV may vary by about 0.01–0.03, but the model's performance simply neither worsens nor improves on the LB, which seems strange. 

Has anyone encountered this?

## Comments and replies

### Comment 1: Rishabh Roy

- Message ID: `3521828`
- Posted: `2026-09-07T09:08:18.460Z`
- Author profile: https://www.kaggle.com/rishabhr0y
- Author type: `COMMENT`
- Competition rank at capture: `650`
- Votes at capture: `-1`

i believe that for me since i am using a point based detector it losing out on lot of features . If i would have used a cell segment based approach it would have al lot of features that we could ave used and identify the tracked cell

### Comment 2: Yubo WANG

- Message ID: `3522082`
- Posted: `2026-09-08T04:03:10.897Z`
- Author profile: https://www.kaggle.com/arksey
- Author type: `COMMENT`
- Competition rank at capture: `924`

I’m seeing pretty much the same behavior.

My motion linker changes give around +0.012 on 5-fold CV on average, with no fold getting worse, but the LB gain is exactly +0.001. I also compared the public 0.946 LB notebook with my older private 0.942 LB notebook under the same 5-fold CV setup, and surprisingly the public one was about 0.015 worse locally. Given that some people have already pointed out possible test-set overfitting in public notebooks, I’m starting to suspect there may also be a fairly large distribution shift between train, public test, and hidden test...?

https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739278

#### Reply 2.1: hengck23

- Message ID: `3522206`
- Posted: `2026-09-08T12:03:12.330Z`
- Author profile: https://www.kaggle.com/hengck23
- Author type: `COMMENT`
- Competition rank at capture: `868`
- Votes at capture: `1`

ask chatgpt "distribution shift between train, public test, and hidden test…" on how to probe or verify.
You definitely can do some online testing/training and send the results back by "smart encoding your submission"

also download more external data or augmentation etc to test locally.
when you evaluate your model on local dataset, don't just look at the numbers. visualise the results and more importantly, explain why (especially statistically). is the error common or will be common?

##### Reply 2.1.1: Yubo WANG

- Message ID: `3522210`
- Posted: `2026-09-08T12:08:33.423Z`
- Author profile: https://www.kaggle.com/arksey
- Author type: `COMMENT`
- Competition rank at capture: `924`

I'm working on finding the cause you mentioned, which was also a problem I encountered when I hit a bottleneck with my ROGII. Thanks a lot🥲

##### Reply 2.1.2: hengck23

- Message ID: `3522221`
- Posted: `2026-09-08T12:50:02.010Z`
- Author profile: https://www.kaggle.com/hengck23
- Author type: `COMMENT`
- Competition rank at capture: `868`
- Votes at capture: `1`

Root cause not necessarily means why it happens. It can also means will it reliably happen. Eg you can make some smart error correction to repair local track. But will this error actually happens in hidden test ( not why it happens).


As a concept example, you find your gain in local lb is due to repair of gap = 5 missing frames. Then you find that of all broken tracks, 80% is gap1, 18% is gap2 … less than one percent is gap5

##### Reply 2.1.3: Yubo WANG

- Message ID: `3522271`
- Posted: `2026-09-08T14:49:59.107Z`
- Author profile: https://www.kaggle.com/arksey
- Author type: `COMMENT`
- Competition rank at capture: `924`

Truly brilliant probe design! Thank you for guiding a Kaggle newcomer like me.

### Comment 3: Rishabh Roy

- Message ID: `3521324`
- Posted: `2026-09-05T14:55:36.760Z`
- Author profile: https://www.kaggle.com/rishabhr0y
- Author type: `COMMENT`
- Competition rank at capture: `650`

on the same boat
