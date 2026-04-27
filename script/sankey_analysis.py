import plotly.graph_objects as go

# Define nodes
nodes = [
    "Initial XAI Papers (818)",      # 0
    "Non-Eval / Other Venues",       # 1 (Discarded)
    "Main Conf. Eval Papers (421)",  # 2
    "Misclassified/False Positives", # 3 (Discarded)
    "Verified Corpus (326)",         # 4
    "Significant Testing (97)",      # 5
    "No Significant Testing (229)"   # 6
]

# Define links (source, target, value)
link_source = [0, 0, 2, 2, 4, 4]
link_target = [1, 2, 3, 4, 5, 6]
link_value  = [397, 421, 80, 326, 97, 244]

# Create figure
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 20,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = nodes,
      color = ["#808080", "#E1E1E1", "#3366CC", "#E1E1E1", "#3366CC", "#109618", "#DC3912"]
    ),
    link = dict(
      source = link_source,
      target = link_target,
      value = link_value,
      color = "rgba(128, 128, 128, 0.2)" # Subtle gray flows
  ))])

fig.update_layout(title_text="Paper Filtering & Verification Pipeline", font_size=12)
# fig.show()

# To save for your paper:
fig.write_image("sankey_flow.pdf")