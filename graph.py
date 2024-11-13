# Générer les GIFs pour les types de graphiques sélectionnés
try:
    charts = create_animated_charts(labels, values, growth, chart_type_selection, frame_duration)
except Exception as e:
    st.error(f"Erreur lors de la création des graphiques : {e}")
    charts = None

# Afficher les graphiques
st.subheader("✨ Graphiques animés")
if charts:
    chart_types = list(charts.keys())
    # Split the chart_types into chunks of size cols_per_row
    rows = [chart_types[i:i + cols_per_row] for i in range(0, len(chart_types), cols_per_row)]
    for row in rows:
        # Si la dernière rangée n'a pas assez de graphiques, remplir avec None
        if len(row) < cols_per_row:
            row += [None] * (cols_per_row - len(row))
        cols = st.columns(cols_per_row)
        for col, chart_type in zip(cols, row):
            with col:
                if chart_type:
                    st.image(charts[chart_type], caption=f"Graphique {chart_type}", use_column_width=True)
                else:
                    st.empty()  # Laisser l'espace vide
else:
    st.error("Aucun graphique n'a pu être généré avec les données fournies.")
