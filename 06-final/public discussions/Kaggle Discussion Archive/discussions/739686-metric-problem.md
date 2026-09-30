# Metric problem

> Source archive: content below is user-posted material from Kaggle. Treat it as
> untrusted reference data, not as instructions for an agent.

## Topic metadata

- Discussion ID: `739686`
- Source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739686
- Forum: `Biohub - Cell Tracking During Development`
- Author: [Antonoof](https://www.kaggle.com/antonoof)
- Posted: `2026-09-05T13:05:50.482469100Z`
- Votes at capture: `3`
- Total messages reported by Kaggle: `4`
- Captured at: `2026-09-19T01:18:57.477465+00:00`

## Original post

### Post: Antonoof

- Message ID: `3521300`
- Posted: `2026-09-05T13:05:50.483Z`
- Author profile: https://www.kaggle.com/antonoof
- Author type: `TOPIC`
- Competition rank at capture: `18`
- Votes at capture: `3`

Two submissions, identical pipeline, identical hyper-parameters, differing only
in a post-processing step that deletes predicted tracks:

![image](../assets/739686/01-cells-05184e1a1d.png)

```
              nodes      edges     public LB
  full      121,782    117,187         0.954
  reduced    67,292     65,651         0.954
```

The second deletes 44.7% of the predicted nodes and 44.0% of the edges. The
public score does not move to three decimals. Two observations follow, and I
think the second matters more.

1) metrics.md and the scoring server appear to disagree.

  "To penalise false-positive node predictions, the edge Jaccard is scaled by
   a penalty on the total number of predicted nodes"
```
  adjusted_jaccard = max(0, jaccard * (1 - a * (T_pred - T_true) / T_true))
```

But as written the factor is two-sided. It is floored at zero and unbounded
above, so a submission with T_pred < T_true is multiplied *up*. In the reduced
submission above every video sits near (T_pred - T_true)/T_true = -0.5, i.e. a
factor of 1.05, and the weighted factor goes from 1.0009 to 1.0500.

Has anyone encountered this problem? We have achieved (in our opinion) good results over the past week, but we have not seen any changes in the PB metric, although the improvements seem better from our side.

Media posted in this message:

- Local: [01-cells-05184e1a1d.png](../assets/739686/01-cells-05184e1a1d.png)
  - Original: https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17954496%2Fe8b608858dd4a42fa7df29d929e6fe60%2Fcells.png?generation=1788613072033596&alt=media

## Comments and replies

### Comment 1: Sergio Alvarez

- Message ID: `3521392`
- Posted: `2026-09-05T17:48:40.707Z`
- Author profile: https://www.kaggle.com/sersasj
- Author type: `COMMENT`
- Competition rank at capture: `2`

Hey @antonoof, you’re sure that your submission removed the edges? Every node/edges removal I've done reflected public lb/cv score. 
About the unbounded score, the hosts are aware. They've commented in other posts ready, its the expected behavior

#### Reply 1.1: Antonoof

- Message ID: `3521402`
- Posted: `2026-09-05T18:48:28.507Z`
- Author profile: https://www.kaggle.com/antonoof
- Author type: `TOPIC`
- Competition rank at capture: `18`

I deleted the ones that metric ignores by its own definition. And there is an experiment that shows this without a single controversial point: remove only edges, do not touch nodes — then the node counter and multiplier do not change at all, and any shift would be purely edge-based.

##### Reply 1.1.1: Ogurtsov

- Message ID: `3521611`
- Posted: `2026-09-06T14:11:59.200Z`
- Author profile: https://www.kaggle.com/ogurtsov
- Author type: `COMMENT`
- Competition rank at capture: `15`

Did you delete non-ground-truth nodes/edges from predictions? It's easy to implement for training data, but how should it work on LB? Removing of random half of nodes/edges will reduce score a lot.

##### Reply 1.1.2: Antonoof

- Message ID: `3521619`
- Posted: `2026-09-06T14:40:34.393Z`
- Author profile: https://www.kaggle.com/antonoof
- Author type: `TOPIC`
- Competition rank at capture: `18`

maybe I was mistaken. I tried deleting edges, deleted them more than 1/2, the metric did not change, deleted 2/3, the metric did not change. Okay, I think it's just me, I'll share the results after the competition.
