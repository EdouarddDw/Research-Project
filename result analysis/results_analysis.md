---
header-includes:
  - \usepackage{graphicx}
---

# Results explanation so everyone is on the same page

## SHAP and NID comparison

### Why do both?

Great question. Quick rundown on both:

- SHAP: how 2 features jointly contribute to a prediction.
- NID: more of a structural measure. It looks at hidden units and infers which inputs are interacting because they are combined strongly through the hidden units.

**In other words:**

**NID** looks at what the model encodes, and **SHAP** looks at how that encoding actually affects predictions on the data.

### Concretely, do they have the same results?

Here is a quick comparison to make things clearer:


\begin{figure}[htbp]
\centering
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/epoch50_NID_fixed.png}
\caption{NID heatmap at epoch 50}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/shap_interactions_epoch_50.png}
\caption{SHAP interaction heatmap at epoch 50}
\end{minipage}
\end{figure}

*Comparison of NID first image and SHAP second image, with the same model, both F3 and both at epoch 50.*

First thing we can see is that both methods found that $x_1$ and $x_2$ have a strong interaction. This is great news, as it is a ground truth interaction.

In the NID heatmap, almost every pair has a fairly large score. Even pairs outside the highlighted ground truth are still strong. The network has encoded a dense interaction structure, not just the true pairs. So NID suggests the model has learned many extra dependencies beyond ground truth.

In the SHAP heatmap, most cells are dark or weak, with only a few brighter regions:

- $x_1$ and $x_2$ very strong
- $x_2$ and $x_9$ strong
- $x_1$ and $x_9$ noticeable
- $x_2$ and $x_4$, $x_4$ and $x_9$ moderate

Here is the thing when looking at ground truth: SHAP only caught $x_1$ and $x_2$, whereas NID caught all of them, although not cleanly.

SHAP is better at isolating the dominant predictive interaction, but it misses some ground truth interactions.  
NID captures the highlighted ground truth interactions, but it also assigns high strength to many non ground truth pairs, so it looks much less specific.

#### Takeaways

This suggests:

- SHAP reflects which interactions matter most for the model’s actual output behavior.
- NID reflects what interaction structure is encoded inside the network.
- The model learned the main true interaction clearly.
- The model may also have learned extra spurious interactions, which NID reveals more strongly than SHAP



