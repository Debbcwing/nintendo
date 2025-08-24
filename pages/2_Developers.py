PAGE_TITLE = "Developers"
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.title("Developer Discovery ⚛️")

st.write("The data displayed in this section is based on a clean dataset that contains 2264 computer games.")

data_path = 'data/deku_dev_df.csv'

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(data_path)

# calculate top-10
@st.cache_data
def top10(data):
    return data.value_counts().nlargest(10).reset_index()

top10_pub = top10(df['publisher'])
top10_list = top10_pub['publisher'].tolist()
top10_df = df[df['publisher'].isin(top10_list)].copy()
top10_df['publisher'] = pd.Categorical(top10_df['publisher'], categories=top10_list, ordered=True)


success_gp = df[df['success']==1]


# color scheme 
colors = px.colors.qualitative.Plotly[:len(top10_list)]
color_map = {publisher: colors[i] for i, publisher in enumerate(top10_list)}

# tabs -----
tab_names = ["Top 10 Publishers", "Game Ratings", "Market Price", "Good-to-know"]
tab1, tab2, tab3, tab4 = st.tabs(tab_names)


with tab1:      # Top 10 Publisher
    st.write(
        ""
    )
    # barplot
    fig_1_bar = px.bar(top10_pub,
                    x='count', y='publisher', orientation='h',
                    title='Top 10 Publishers by Game Count',
                    labels={'publisher': 'Publisher', 'count': 'Number of Games'},
                    category_orders={'publisher': top10_list},
                    color='publisher', color_discrete_map=color_map)
    fig_1_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_1_bar)
    # scatterplot
    fig_1_scatter = px.scatter(top10_df, x='avg_score', y='msrp_price', color='publisher',
                            color_discrete_map=color_map, hover_data=['title'], 
                            title='Market price to user ratings by Publisher',
                            category_orders={'publisher': top10_list},
                            labels={'avg_score': 'Game Ratings', 'msrp_price': 'Market Price (USD)'})
    fig_1_scatter.update_layout(legend_title='Publisher')
    st.plotly_chart(fig_1_scatter)

with tab2:      # Game Ratings
    st.write(
        "The ratings are based on the average scores from game critics and users."
    )
    st.write(
        "Choose one variable:", key='tab2'
    )
    if st.checkbox("Beat Time"):
        fig_2a_scatter = px.scatter(df, x='hltb_main_story', y='avg_score', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='Game Ratings to Game Beat Time',
                                labels={'avg_score': 'Average Score', 'hltb_main_story': 'Game Beat Time (Hours)'})
        fig_2a_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_2a_scatter)
    if st.checkbox("Game Size"):
        fig_2_scatter = px.scatter(df, x='download_size', y='avg_score', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='Game Ratings to Game Size',
                                labels={'avg_score': 'Average Score', 'download_size': 'Download Size (GB)'})
        fig_2_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_2_scatter)
    if st.checkbox("Number of Available Language"):
        fig_2b_scatter = px.scatter(df, x='lang_count', y='avg_score', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='Game Ratings to Number of Available Language',
                                labels={'avg_score': 'Average Score', 'lang_count': 'Number of Languages'})
        fig_2b_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_2b_scatter)
    if st.checkbox("Number of Available Platform"):
        fig_2c_scatter = px.scatter(df, x='plat_count', y='avg_score', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='Game Ratings to Number of Game Platform',
                                labels={'avg_score': 'Average Score', 'plat_count': 'Number of Platforms'})
        fig_2c_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_2c_scatter)

with tab3:      # Market Price
    st.write(
        "The prices are based on the listed price of the game."
    )
    st.write(
        "Choose one variable:", key='tab3'
    ) 
    if st.checkbox("Beat Time", key='tab3a'):
        fig_3a_scatter = px.scatter(df, x='hltb_main_story', y='msrp_price', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='User Ratings to Game Beat Time',
                                labels={'msrp_price': 'Market Price (USD)', 'hltb_main_story': 'Game Beat Time (Hours)'})
        fig_3a_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_3a_scatter)

    if st.checkbox("Game Size", key='tab3'):
        fig_3_scatter = px.scatter(df, x='download_size', y='msrp_price', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='Market Price to Game Size',
                                labels={'msrp_price': 'Market Price (USD)', 'download_size': 'Download Size (GB)'})
        fig_3_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_3_scatter)

    if st.checkbox("Number of Available Language", key='tab3b'):
        fig_3b_scatter = px.scatter(df, x='lang_count', y='msrp_price', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='Market Price to Number of Available Language',
                                labels={'msrp_price': 'Market Price (USD)', 'lang_count': 'Number of Languages'})
        fig_3b_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_3b_scatter)
    if st.checkbox("Number of Available Platform", key='tab3c'):
        fig_3c_scatter = px.scatter(df, x='plat_count', y='msrp_price', color='success',
                                color_continuous_scale='RdBu', hover_data=['title'], 
                                title='Market Price to Number of Game Platform',
                                labels={'msrp_price': 'Market Price (USD)', 'plat_count': 'Number of Platforms'})
        fig_3c_scatter.update_layout(legend_title='Publisher')
        st.plotly_chart(fig_3c_scatter)

with tab4:  # Good-to-know
    st.write(
        "These key trends can be useful for developing the most successful and profitable game! ***The Success*** group is defined as those with ratings over 70."
    )
    st.write("")
    st.write("")
    st.write(
        "🍯 Find the Safe Range / Sweet Spot for:"
    )
    if st.checkbox("Game Beat Time"):
        samples = tab4.radio("Samples", ["Overall", "The Success"])
        if samples == "Overall":
            const = 13.6947
            beta1 = 1.1430  # model coef
            beta2 = -0.0105 # model coef
            optimal_beat_time = -beta1 / (2 * beta2)
            beat_time_range = np.linspace(df['hltb_main_story'].min(), df['hltb_main_story'].max(), 50)
            price_pred = const + beta1*beat_time_range + beta2*beat_time_range**2

            fig_4 = go.Figure()
            fig_4.add_trace(go.Scatter(x=df['hltb_main_story'], y=df['msrp_price'], mode='markers', name='Game Data',
                            marker=dict(color='green', size=6),
                            customdata=np.stack([df['title']], axis=-1), 
                            hovertemplate='<b>%{customdata[0]}</b><br>' +  
                                          'Beat Time: %{x} hrs<br>' +
                                          'Price: $%{y}<extra></extra>'))
            fig_4.add_trace(go.Scatter(x=beat_time_range, y=price_pred, mode='lines', name='Fitted Curve', line=dict(color='blue', width=5, dash='solid')))
            fig_4.add_trace(go.Scatter(x=[optimal_beat_time], y=[const + beta1*optimal_beat_time + beta2*optimal_beat_time**2],
                                    mode='markers', name='The Sweet Spot', marker=dict(color='red', size=12),
                                    hovertemplate= 'Beat Time: %{x} hrs<br>' + 'Price: $%{y}<extra></extra>'))

            fig_4.update_layout(title='Market Price vs Game Beat Time', xaxis_title='Game Beat Time (Hours)',
                                yaxis_title='Market Price (USD)')
            st.plotly_chart(fig_4)
        else:       # only success group samples
            const = 13.3381
            beta1 = 1.1403 
            beta2 = -0.0096
            optimal_beat_time = -beta1 / (2 * beta2)
            beat_time_range = np.linspace(success_gp['hltb_main_story'].min(), success_gp['hltb_main_story'].max(), 100)
            price_pred = const + beta1*beat_time_range + beta2*beat_time_range**2

            fig_4b = go.Figure()
            fig_4b.add_trace(go.Scatter(x=df['hltb_main_story'], y=df['msrp_price'], mode='markers', name='The Success',
                            marker=dict(color='green', size=6),
                            customdata=np.stack([df['title']], axis=-1), 
                            hovertemplate='<b>%{customdata[0]}</b><br>' +  
                                          'Beat Time: %{x} hrs<br>' +
                                          'Price: $%{y}<extra></extra>'))
            fig_4b.add_trace(go.Scatter(x=beat_time_range, y=price_pred, mode='lines', name='Fitted Curve', line=dict(color='blue', width=5, dash='solid')))
            fig_4b.add_trace(go.Scatter(x=[optimal_beat_time], y=[const + beta1*optimal_beat_time + beta2*optimal_beat_time**2],
                                    mode='markers', name='The Sweet Spot', marker=dict(color='red', size=12),
                                    hovertemplate= 'Beat Time: %{x} hrs<br>' + 'Price: $%{y}<extra></extra>'))

            fig_4b.update_layout(title='Market Price vs Game Beat Time', xaxis_title='Game Beat Time (Hours)',
                                yaxis_title='Market Price (USD)')
            st.plotly_chart(fig_4b)

    if st.checkbox("Game Size", key='tab4b'):
        samples = tab4.radio("Samples", ["Overall", "The Success"], key='radio4b')
        if samples == "Overall":
            const = 13.7388
            beta1 = 3.7379  # model coef
            beta2 = -0.1001 # model coef
            optimal_size = -beta1 / (2 * beta2)
            size_range = np.linspace(df['download_size'].min(), df['download_size'].max(), 100)
            price_pred = const + beta1*size_range + beta2*size_range**2

            fig_4c = go.Figure()
            fig_4c.add_trace(go.Scatter(x=df['download_size'], y=df['msrp_price'], mode='markers', name='Game Data',
                            marker=dict(color='green', size=6),
                            customdata=np.stack([df['title']], axis=-1), 
                            hovertemplate='<b>%{customdata[0]}</b><br>' +  
                                          'Beat Time: %{x} hrs<br>' +
                                          'Price: $%{y}<extra></extra>'))
            fig_4c.add_trace(go.Scatter(x=size_range, y=price_pred, mode='lines', name='Fitted Curve', line=dict(color='blue', width=5, dash='solid')))
            fig_4c.add_trace(go.Scatter(x=[optimal_size], y=[const + beta1*optimal_size + beta2*optimal_size**2],
                                        mode='markers', name='The Sweet Spot', marker=dict(color='red', size=12),
                                        hovertemplate= 'Download Size: %{x} GB<br>' + 'Price: $%{y}<extra></extra>'))

            fig_4c.update_layout(title='Market Price vs Game Size', xaxis_title='Game Size (GB)',
                                yaxis_title='Market Price (USD)')
            st.plotly_chart(fig_4c)
        else:       # only success group samples
            const = 14.2731
            beta1 = 3.8481 
            beta2 = -0.1006
            optimal_size = -beta1 / (2 * beta2)
            size_range = np.linspace(df['download_size'].min(), df['download_size'].max(), 100)
            price_pred = const + beta1*size_range + beta2*size_range**2

            fig_4d = go.Figure()
            fig_4d.add_trace(go.Scatter(x=success_gp['download_size'], y=success_gp['msrp_price'], mode='markers', name='Game Data',
                            marker=dict(color='green', size=6),
                            customdata=np.stack([df['title']], axis=-1), 
                            hovertemplate='<b>%{customdata[0]}</b><br>' +  
                                          'Beat Time: %{x} hrs<br>' +
                                          'Price: $%{y}<extra></extra>'))
            fig_4d.add_trace(go.Scatter(x=size_range, y=price_pred, mode='lines', name='Fitted Curve', line=dict(color='blue', width=5, dash='solid')))
            fig_4d.add_trace(go.Scatter(x=[optimal_size], y=[const + beta1*optimal_size + beta2*optimal_size**2],
                                        mode='markers', name='The Sweet Spot', marker=dict(color='red', size=12),
                                        hovertemplate= 'Download Size: %{x} GB<br>' + 'Price: $%{y}<extra></extra>'))

            fig_4d.update_layout(title='Market Price vs Game Size', xaxis_title='Game Size (GB)',
                                yaxis_title='Market Price (USD)')
            st.plotly_chart(fig_4d)
    
    
    ## another section
    st.write(
        "Find a reference for potential price based on:"
    )    
    if st.checkbox("Game Ratings", key='tab4e'):
        const = 13.0822
        beta = 0.1496  # model coef
        score_range = np.linspace(df['avg_score'].min(), df['avg_score'].max(), 100)
        price_pred = const + beta * score_range
        const_succ = -13.9619
        beta_succ = 0.4971
        score_range_succ = np.linspace(success_gp['avg_score'].min(), success_gp['avg_score'].max(), 50)
        price_pred_succ = const_succ + beta_succ * score_range_succ

        fig_4e = go.Figure()
        fig_4e.add_trace(go.Scatter(x=df['avg_score'], y=df['msrp_price'], mode='markers', name='Game Data', opacity=0.7,
                        marker=dict(color='green', size=6),
                        customdata=np.stack([df['title']], axis=-1), 
                        hovertemplate='<b>%{customdata[0]}</b><br>' +  
                                        'Beat Time: %{x} hrs<br>' +
                                        'Price: $%{y}<extra></extra>'))
        fig_4e.add_trace(go.Scatter(x=score_range, y=price_pred, mode='lines', name='Fitted Curve for All', 
                                    line=dict(color='blue', width=5, dash='solid')))
        fig_4e.add_trace(go.Scatter(x=score_range_succ, y=price_pred_succ, mode='lines', name='Fitted Curve for The Success', 
                                    line=dict(color='red', width=5, dash='solid')))
        fig_4e.update_layout(title='Market Price vs Game Ratings', xaxis_title='Game Ratings',
                            yaxis_title='Market Price (USD)')
        st.plotly_chart(fig_4e)
