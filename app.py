import dash
from dash import html, dcc, callback, ctx
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import us
import plotly.express as px
from dash import dash_table  # For the Records tab data table

# ------------------------------------------------------------------
# 1. Read CSV and prep data
# ------------------------------------------------------------------
df = pd.read_csv("sales_data_sample_Orders.csv")

# Convert Revenue to numeric
df["Revenue"] = pd.to_numeric(df["Revenue"].replace('[\\$,]', '', regex=True), errors="coerce")

df["Profit"] = df["Profit"].replace([r'\$', r','], regex = True, value = '')
df["Profit"] = df["Profit"].apply(lambda x:- float(x[1:-1]) if x.startswith("(") and x.endswith(")") else float(x))
print(df["Profit"])

# Convert Order Date to datetime, then create Year & Month columns
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# Aggregate monthly totals for KPI sparklines
df_monthly = (
    df.groupby(df["Order Date"].dt.month)
      .agg({
          "Revenue": "sum",
          "Order ID": "nunique",
          "Customer ID": "nunique",
          "Category": "nunique"
      })
      .rename(columns={
          "Order ID": "Orders",
          "Customer ID": "Customers"
      })
      .sort_index()
)

# For the year dropdown, show unique years (descending order)
year_options = sorted(df["Year"].unique(), reverse=True)
year_options = year_options[:3]
print(year_options)

# ------------------------------------------------------------------
# 2. Initialize the Dash app
# ------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# ------------------------------------------------------------------
# 3. Top Navbar (with Dashboard links + Year dropdown)
# ------------------------------------------------------------------
top_nav = dbc.Navbar(
    dbc.Container([
        html.Img(
            src="/assets/upper_left.png",
            style={
                "height": "40px",  
                "margin-right": "15px" 
            }
        ),
        dbc.Nav(
            [
                dbc.NavItem(
                    dbc.NavLink([
                        html.Div(
                            [
                                html.Div(
                                    style={
                                        'background-color': 'white', 
                                        'width': '4px', 
                                        'height': '4px',
                                        'border': '1px solid white'
                                    }
                                ) for _ in range(4)
                            ],
                                    style={
                                        'display': 'grid', 
                                        'grid-template-columns': '1fr 1fr', 
                                        'gap': '1px', 
                                        'width': '12px', 
                                        'height': '12px'
                                    },
                                    className="me-2"
                                ),
                            "Dashboard"
                            ], active=True, href="#", className = "bg-dark text-white rounded d-flex align-items-center gap-2", id = "nav-dashboard")),
                dbc.NavItem(dbc.NavLink("Insights", href="#", className = "text-dark", id = "nav-insight")),
                dbc.NavItem(dbc.NavLink("Records", href="#", className = "text-dark", id = 'nav-records')),
            ],
            pills=True,
            navbar=True
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavItem(
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[{"label": str(y), "value": str(y)} for y in year_options],
                            value=str(year_options[0]),  # default to most recent
                            clearable=False,
                            style={"width": "120px"}
                        )
                    )
                ],
                className="ms-auto",
                navbar=True
            ),
            id="navbar-collapse",
            navbar=True
        ),
    ], fluid=True),
    color="white",
    dark=True,
    className="shadow-sm"
)

# ------------------------------------------------------------------
# 4. Header (white) with title, metric dropdown, export button
# ------------------------------------------------------------------
header = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Sales Dashboard", className="display-5 fw-bold"),
            html.P("Access a comprehensive summary of KPIs and vital insights."),
        ], width=6),

        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id="metric-dropdown",
                        options=[
                            {"label": "Revenue", "value": "Revenue"},
                            {"label": "Quantity", "value": "Quantity"},
                            {"label": "Profit", "value": "Profit"},
                            {"label": "Customers", "value": "Customers"},
                        ],
                        value="Revenue",
                        clearable=False,
                        style={"width": "150px"}
                    ),
                    width="auto"
                ),
                dbc.Col(
                    dbc.Button("Export", color="primary", className="px-4"),
                    width="auto"
                ),
            ],
            justify="end",
            align="center",
            className="g-2"
            )
        ], width=6, className="d-flex align-items-center justify-content-end"),
    ], className="py-3"),
], fluid=True, className="bg-white shadow-sm")

# ------------------------------------------------------------------
# 5. Function to create bar chart
#    - Takes selected metric and year
#    - Plots current year as bars, previous year as horizontal dashes
# ------------------------------------------------------------------
def make_sparkline(current_data, previous_data, selected_year, previous_year, metric):
    """Return a minimal sparkline figure with the given y data."""
    months = ["January", "Febuary", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    selected_indices = [1, 3, 5, 7, 9, 11]
    selected_labels = ["F", "A", "J", "A", "O", "D"]
    percent_change = [((current-prev)/prev)*100 for current, prev in zip(current_data, previous_data)]
    arrow = ['▲' if pct > 0 else '▼' for pct in percent_change]
    color = ['green' if pct>0 else 'red' for pct in percent_change]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
        x=list(range(len(previous_data))),
            y=previous_data,
            mode='lines',
            fill= 'tozeroy',
            line=dict(color='#A7C7E7'),
            showlegend= False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(current_data))),
            y=current_data,
            mode='lines',
            line=dict(color='#1c4e8b', width = 3),
            showlegend= False,
            hovertemplate= (
                f"<br>"    f"<b style='font-size: 11px;'>%{{text}}</b><br>"  
                f"<span style='color:%{{customdata[3]}};'>"    
                f"%{{customdata[2]}}%{{customdata[0]:.2f}} %</span><br>"      
                "_________________<br>"   
                f"<span style = 'font-size: 10px'>{selected_year} {metric}: <b>%{{y:,}}</b></span><br>"  
                # Bold current year value
                f"<span style = 'font-size: 10px'>{previous_year} {metric}: <b>%{{customdata[1]:}}</b></span><br>"  
                # Bold previous year value     
                #"_________________<br>""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"  
                # Adds additional spaces after content (you can adjust number of &nbsp; for more space)
                "<extra></extra>"),
                customdata=list(zip(percent_change, previous_data, arrow, color)),
                text = months
            )
        ),
    
    fig.update_layout(
        margin=dict(l=0, r=50, t=0, b=0),
        paper_bgcolor="rgba(255,255,255,0.8)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode = "x unified",
        #xaxis_visible=False,
        yaxis_visible=False,
        height=100,
        width = 145,
        xaxis = dict(
            tickmode = "array",
            tickvals = selected_indices,
            ticktext = selected_labels,
            zeroline = False,
            showgrid = False
        ),
        showlegend = False
    )
    return fig

def format_metric_val(metric, value):
    if metric in ["Revenue", "Profit"]:
        return f"${float(value):,.0f}"
    else:
        return f"{float(value):,.0f}"


def create_bar_chart(metric="Revenue", selected_year=None):
    # Ensure 'Order Date' is datetime format
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Extract Year and Month from the Date
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month

    # If no selected_year is provided, use the latest year available
    if selected_year is None:
        selected_year = df["Year"].max()
    else:
        selected_year = int(selected_year)

    previous_year = selected_year - 1

    if metric == 'Customers':
        df_current_year = df[df["Year"] == selected_year].groupby("Month", as_index=False)["Customer ID"].nunique().rename(columns = {"Customer ID": f"{metric}_current"})
        df_previous_year = df[df["Year"] == previous_year].groupby("Month", as_index=False)["Customer ID"].nunique().rename(columns = {"Customer ID": f"{metric}_previous"})
    else:
        # Aggregate revenue by month for the latest year
        df_current_year = df[df["Year"] == selected_year].groupby("Month", as_index=False)[metric].sum()

        # Aggregate revenue by month for the previous year
        df_previous_year = df[df["Year"] == previous_year].groupby("Month", as_index=False)[metric].sum()

    # Merge to ensure both current and previous year data align by month
    df_agg = pd.merge(df_current_year, df_previous_year, on="Month", suffixes=("_current", "_previous"), how="left")

    # Define month abbreviations for display
    month_abbr = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                  7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    df_agg["Month"] = df_agg["Month"].map(month_abbr)

    #max value month
    max_val_month = df_agg.loc[df_agg[f"{metric}_current"].idxmax()]
    #min val month
    min_val_month = df_agg.loc[df_agg[f"{metric}_current"].idxmin()]
    colors = ["#4c7cf5" if month == max_val_month["Month"] else "#a4c2e0" for month in df_agg["Month"]]

    # Create the bar chart
    df_agg["% Change"] = ((df_agg[f"{metric}_current"] - df_agg[f"{metric}_previous"]) / df_agg[f"{metric}_previous"]) * 100

    df_agg["colors"] = df_agg["% Change"].apply(lambda x: 'green' if x>0 else 'red')
    df_agg["arrow"] = df_agg["% Change"].apply(lambda x: '▲' if x>0 else '▼')

    custom_data = df_agg[[f"{metric}_previous", f"{metric}_current", "% Change", "arrow", "colors"]].apply(
        lambda
             row: [
                 format_metric_val(metric,row[f"{metric}_previous"]),
                 format_metric_val(metric,row[f"{metric}_current"]),
                 row["% Change"],
                 row["arrow"],
                 row["colors"]
                 
             ], axis = 1).values
    
    avgval = df_agg[f"{metric}_current"].mean()
    fig = go.Figure()
    
    # Add bars for the current year
    fig.add_trace(
        go.Bar(
            x=df_agg["Month"],
            y=df_agg[f"{metric}_current"],
            
            name="Current Year",
            marker_color=colors,
            hoverlabel= dict(
                bgcolor = "white",
                font_size = 12,

            ),
            hovertemplate=(    
                f"<br><br>"    f"<b style='font-size: 14px;'>%{{x}}</b><br><br>"  
                # Increase font size of the month   
                f"<span style='color:%{{customdata[4]}}; font-weight:bold; font-size:15px;'>"    
                f"%{{customdata[3]}}%{{customdata[2]:.2f}} %</span><br>"      
                "_________________<br><br>"    
                f"{selected_year} {metric}: <b>%{{customdata[1]}}</b><br>"  
                # Bold current year value
                f"{previous_year} {metric}: <b>%{{customdata[0]}}</b><br>"  
                # Bold previous year value     
                "_________________<br>""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"  
                # Adds additional spaces after content (you can adjust number of &nbsp; for more space)
                "<extra></extra>"),
 
            customdata=custom_data
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [df_agg["Month"].iloc[0], df_agg["Month"].iloc[-1]],
            y = [avgval] * len(df_agg["Month"]),
            mode = "lines", 
            line = dict(color = "lightgrey", width = 2, dash = "dash"),
            name = "Average Line",
            hoverinfo = "text",
            hovertemplate=(        
                            "<br><br>"  # Extra spacing for a larger box        
                           "<b style='font-size:12px;'> Avg. " + metric + "</b><br>"  # Larger font         
                           #"________________________<br><br>"        
                           "<b style='font-size:12px; color:#4c7cf5;'>" + str(round(avgval, 2)) + "</b><br>"                        
                           #"________________________<br><br>"                        
                           #"<extra></extra>"  # Removes default trace name from hover
                           ),
            hoverlabel=dict(    bgcolor="white",  
            # Set background color to white    
            font_size=12,  
            # Adjust font size  
            font_family="Arial",  
            # Set font family    
            ),
        )    
    )

    # Add horizontal dashes for the previous year
    fig.add_trace(
        go.Scatter(
            x=df_agg["Month"],
            y=df_agg[f"{metric}_previous"],
            name="Previous Year",
            mode="markers",
            marker=dict(
                symbol="line-ew",  # Horizontal dash marker
                color="black",
                size=15,  # Adjust length of dashes
                line_width=3  # Adjust thickness
            )
        )
    )

    #add notation for min bar
    fig.add_trace(
        go.Scatter(
            x = [max_val_month["Month"]],
            y = [max_val_month[f"{metric}_current"]],
            mode = "text", 
            text = format_metric_val(metric, float(max_val_month[f"{metric}_current"])),
            textposition = "top center",
            showlegend = False
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [min_val_month["Month"]],
            y = [min_val_month[f"{metric}_current"] + 1.25],
            mode = "text", 
            text = format_metric_val(metric, float(min_val_month[f"{metric}_current"])),
            textposition = "top center",
            showlegend = False
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Months of {selected_year} - {metric}",
        xaxis_title="Month",
        yaxis_title=metric,
        template="plotly_white",
        barmode="group",
        xaxis=dict(tickangle=0),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )

    return fig

# ------------------------------------------------------------------
# function to create horizontal bar charts for Category and Region
# ------------------------------------------------------------------
#finish
def create_horizontal_bar(data, title, selected_metric):
    """Create horizontal bar chart for categories or regions"""
    # Create a simple horizontal bar chart
    fig = go.Figure()

    blocks = []

    
    for _, row in data.iterrows():
        if title == "Category":
            value = row['value']
            val_col = 'value'
            margin_bot1 = "38px"
            margin_bot2 = "12px"
        label = row[title]
        change = row["% Change"]
        color = 'green' if change >= 0 else 'red'
        arrow = '▲' if change >= 0 else '▼'
        blocks.append(html.Div([
            html.Div([
                html.Span(f'{label}'),
                html.Span(f'{arrow} {abs(round(change,2))}%', style={"color": color, "fontSize": "14px"})
            ],
            style={"marginBottom": margin_bot2}),
            html.Div([
                dbc.Progress(value = value, max = data[val_col].max()*1.2, color="primary",             
                             style={"height": "10px", "flex": "1", "backgroundColor": "transparent","borderRadius": "0px" }),
                             html.Div(
                                 f'{format_metric_val(selected_metric, value)}', style = {"color": "blue", "fontSize": "14", "marginLeft": "10px"}
                             )
            ], style = {"display": "flex", "alignItems":"center", "gap": "8px"})
            
            
        ])) 
    return html.Div([html.H5(title, style = {"fontweight": "bold", "marginBottom": "25px"}), 
                     *blocks], style={"padding": "10px", "border": "1px solid #eee", "borderRadius": "10px", "width": "300px"})
    
    
    
def convert_state_map_names(df):
    state_abbr_map = {state.name:state.abbr for state in us.states.STATES}
    df["State"] = df["State"].map(state_abbr_map)
    return df

def calc_percentage_change(current, previous):
        return ((current - previous) / previous)*100 if previous != 0 else 0

# ------------------------------------------------------------------
# Function to create US state map
# ------------------------------------------------------------------
def create_state_map(df, py_df, selected_year, metric):
    df = convert_state_map_names(df)
    py_df = convert_state_map_names(py_df)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    py_df[metric] = pd.to_numeric(py_df[metric], errors="coerce")

    merge_df = pd.merge(df[['State', metric]], py_df[['State', metric]], on = 'State', how = 'outer', suffixes = ('_current', '_previous')).fillna(0)
    merge_df["% Change"] = merge_df.apply(lambda row: calc_percentage_change(row[f'{metric}_current'], row[f'{metric}_previous'])
    if row[f'{metric}_current'] != 0 and row[f'{metric}_previous'] != 0 else None, axis = 1)
    
    merge_df["colors"] = merge_df["% Change"].apply(lambda x: 'green' if x>0 else 'red')
    merge_df["arrow"] = merge_df["% Change"].apply(lambda x: '▲' if x>0 else '▼')
    
    #merge_df["State full name"] = merge_df["State"].apply(lambda abbr:us.states.lookup(abbr).name if us.states.lookup(abbr) else abbr)
    #merge_df = merge_df[merge_df["State"].astype(str) != '0']
    merge_df['State'] = merge_df['State'].astype(str).str.strip()
    merge_df= merge_df[merge_df['State'] != '0']
    merge_df["State full name"] = merge_df["State"].apply(lambda abbr:us.states.lookup(abbr).name if us.states.lookup(abbr) else abbr)
    print(merge_df)

    custom_data2 = merge_df.apply(
        lambda
          row: [
                 format_metric_val(metric,row[f"{metric}_previous"]) if pd.notna(row[f"{metric}_previous"]) else 'N/A',
                 format_metric_val(metric,row[f"{metric}_current"]) if pd.notna(row[f"{metric}_current"]) else 'N/A',
                 f"{row["% Change"]:.2f}%" if pd.notna(row["% Change"]) else " ",
                 row["arrow"],
                 row["colors"],
                 row["State full name"]
                 
             ], axis = 1).values
    if metric=="Customers":
        color_column = "Customer ID"
    else:
        color_column = metric
    
    avgval = merge_df[f"{metric}_current"].mean()
    fig = go.Figure()

    fig = px.choropleth(
        merge_df,
        locations="State",
        locationmode="USA-states",
        color = f'{color_column}_current',
        hover_data= {f'{color_column}_current':True, 'State':True},
        color_continuous_scale= "Blues",
        scope="usa",
    )

    fig.update_traces(
        marker_line_color = "white",
        hoverlabel = dict(
            bgcolor = "rgba(255, 255, 255, .7)",
            font = dict(
                color = "black",
            )
        ),
    

    # hovertemplate='%{location}<br>' + metric + ': %{z}',  # Custom hover information
    hovertemplate=(
        "<br><br>"
        "<b style='font-size: 14px;'>%{customdata[5]}</b><br><br>"  # State name
        "%{customdata[3]} <span style='color:%{customdata[4]}; font-weight:bold; font-size:15px;'>"
        "%{customdata[2]}</span><br>" 
        "_________________<br><br>" 
        f"{selected_year} {metric}: <b>%{{customdata[1]}}</b><br>"  # Current year value
        f"{int(selected_year) - 1} {metric}: <b>%{{customdata[0]}}</b><br>"  # Previous year value
        "_________________<br>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"  # Adds additional spaces
        "<extra></extra>"
    ),
    customdata=custom_data2

),

    
    


    fig.update_layout(
        #title_text=f"Revenue by State ({selected_year})",
        geo=dict(
            scope="usa",
            projection=go.layout.geo.Projection(type="albers usa"),
            showlakes=True,
            lakecolor="rgb(255, 255, 255)"
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=300,
        coloraxis_showscale = False

    )
    
    # Return the additional figures
    return fig

# Helper function to create horizontal bar charts with reference lines
def create_horizontal_bar_with_reference(df, x_col, y_col, title, show_values=False, is_currency=False, color_highest=False):
    """
    Create a horizontal bar chart with reference lines matching the style shown in the image
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis values
        y_col: Column name for y-axis labels
        title: Chart title
        show_values: Whether to show values at the end of bars
        is_currency: Whether to format values as currency
        color_highest: Whether to color the highest value bar differently
    """
    # Sort the dataframe by the x_col in descending order
    df = df.sort_values(by=x_col, ascending=False)
    
    # Create a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    # Get the average value for the reference line
    avg_value = plot_df[x_col].mean()
    
    # Create colors list - highlight the highest value if requested
    if color_highest:
        colors = ['#4662D9' if i == 0 else '#A7C7E7' for i in range(len(plot_df))]
    else:
        colors = ['#A7C7E7'] * len(plot_df)
    
    # Create the horizontal bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=plot_df[x_col],
        y=plot_df[y_col],
        orientation='h',
        marker_color=colors,
        width=0.6,  # Make bars thinner
    ))
    
    # Add reference lines (vertical black lines) at specific positions
    for i, row in plot_df.iterrows():
        # Add a vertical line at a position that's about 70% of the max value
        reference_position = 0.7 * plot_df[x_col].max()
        
        fig.add_shape(
            type="line",
            x0=reference_position,
            x1=reference_position,
            y0=i - 0.25,  # Adjust to align with the bar
            y1=i + 0.25,  # Adjust to align with the bar
            line=dict(color="black", width=2),
        )
    
    # Add value labels at the end of bars if requested
    if show_values:
        for i, row in plot_df.iterrows():
            value = row[x_col]
            if is_currency:
                value_text = f"${value:,.2f}"
            else:
                value_text = f"{int(value)}" if isinstance(value, (int, float)) and value.is_integer() else f"{value:,.1f}"
            
            fig.add_annotation(
                x=value,
                y=row[y_col],
                text=value_text,
                showarrow=False,
                xshift=5,
                align="left",
                font=dict(color="#4662D9" if i == 0 and color_highest else "black")
            )
    
    # Update layout for clean appearance
    fig.update_layout(
        title=title,
        title_font=dict(size=14),
        margin=dict(l=0, r=40, t=30, b=10),  # Increased right margin for value labels
        height=250,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",  # Reverse y-axis to match your image
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    
    return fig

# Create a function to generate the Insights tab content
def create_insights_tab(df, selected_year):
    """
    Creates the Insights tab content with Top 5 States and Segments charts
    
    Args:
        df: DataFrame with sales data
        selected_year: Selected year for filtering
    
    Returns:
        A Dash component representing the Insights tab
    """
    # Filter data for selected year
    df_year = df[df["Year"] == int(selected_year)]
    
    # -----------------------------------------------------------------
    # State-level aggregations for Top 5 States charts
    # -----------------------------------------------------------------
    # Quantity by State
    df_state_qty = df_year.groupby("State", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False).head(5)
    
    # Revenue by State
    df_state_rev = df_year.groupby("State", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False).head(5)
    
    # Profit by State
    df_state_profit = df_year.groupby("State", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False).head(5)
    
    # Customers by State
    df_state_cust = df_year.groupby("State", as_index=False)["Customer ID"].nunique().rename(
        columns={"Customer ID": "Customers"}
    ).sort_values("Customers", ascending=False).head(5)
    
    # -----------------------------------------------------------------
    # Segment-level aggregations (using existing Segment column)
    # -----------------------------------------------------------------
    # Quantity by Segment
    df_seg_qty = df_year.groupby("Segment", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
    
    # Revenue by Segment
    df_seg_rev = df_year.groupby("Segment", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    
    # Profit by Segment
    df_seg_profit = df_year.groupby("Segment", as_index=False)["Profit"].sum().sort_values("Profit", ascending=False)
    
    # Customers by Segment
    df_seg_cust = df_year.groupby("Segment", as_index=False)["Customer ID"].nunique().rename(
        columns={"Customer ID": "Customers"}
    ).sort_values("Customers", ascending=False)
    
    # -----------------------------------------------------------------
    # Create charts for Top 5 States
    # -----------------------------------------------------------------
    # Quantity by State
    fig_state_qty = create_horizontal_bar_with_reference(
        df_state_qty, 
        x_col="Quantity", 
        y_col="State", 
        title="Top 5 States by Quantity",
        show_values=True,
        color_highest=True
    )
    
    # Revenue by State
    fig_state_rev = create_horizontal_bar_with_reference(
        df_state_rev, 
        x_col="Revenue", 
        y_col="State", 
        title="Top 5 States by Revenue",
        show_values=True,
        is_currency=True,
        color_highest=True
    )
    
    # Profit by State
    fig_state_profit = create_horizontal_bar_with_reference(
        df_state_profit, 
        x_col="Profit", 
        y_col="State", 
        title="Top 5 States by Profit",
        show_values=True,
        is_currency=True,
        color_highest=True
    )
    
    # Customers by State
    fig_state_cust = create_horizontal_bar_with_reference(
        df_state_cust, 
        x_col="Customers", 
        y_col="State", 
        title="Top 5 States by Customers",
        show_values=True,
        color_highest=True
    )
    
    # -----------------------------------------------------------------
    # Create charts for Segments
    # -----------------------------------------------------------------
    # Quantity by Segment
    fig_seg_qty = create_horizontal_bar_with_reference(
        df_seg_qty, 
        x_col="Quantity", 
        y_col="Segment", 
        title="Segments by Quantity",
        show_values=True,
        color_highest=True
    )
    
    # Revenue by Segment
    fig_seg_rev = create_horizontal_bar_with_reference(
        df_seg_rev, 
        x_col="Revenue", 
        y_col="Segment", 
        title="Segments by Revenue",
        show_values=True,
        is_currency=True,
        color_highest=True
    )
    
    # Profit by Segment
    fig_seg_profit = create_horizontal_bar_with_reference(
        df_seg_profit, 
        x_col="Profit", 
        y_col="Segment", 
        title="Segments by Profit",
        show_values=True,
        is_currency=True,
        color_highest=True
    )
    
    # Customers by Segment
    fig_seg_cust = create_horizontal_bar_with_reference(
        df_seg_cust, 
        x_col="Customers", 
        y_col="Segment", 
        title="Segments by Customers",
        show_values=True,
        color_highest=True
    )
    
    # -----------------------------------------------------------------
    # Create the layout with two rows of charts
    # -----------------------------------------------------------------
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Insights", className="display-5 fw-bold"),
                html.P("Access a detailed breakdown of metrics by states, segments, and ship modes."),
            ], width=12),
        ], className="py-3"),
        
        # First row: Top 5 States charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_state_qty, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
            dbc.Col([
                dcc.Graph(figure=fig_state_rev, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
            dbc.Col([
                dcc.Graph(figure=fig_state_profit, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
            dbc.Col([
                dcc.Graph(figure=fig_state_cust, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
        ], className="mb-4"),
        
        # Second row: Segments charts
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_seg_qty, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
            dbc.Col([
                dcc.Graph(figure=fig_seg_rev, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
            dbc.Col([
                dcc.Graph(figure=fig_seg_profit, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
            dbc.Col([
                dcc.Graph(figure=fig_seg_cust, config={"displayModeBar": False}, style={"height": "250px"})
            ], width=3),
        ]),
    ], fluid=True)

# Function to generate the records tab content
# def generate_records_tab(selected_year):
#     """Generate the Records tab content"""
#     df_year = df[df["Year"] == int(selected_year)]
    
#     # Create a copy with formatted columns for display
#     df_display = df_year.copy()
    
#     # Format currency columns
#     for col in ['Revenue', 'Profit']:
#         if col in df_display.columns:
#             df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
#     # Format date columns
#     for col in ['Order Date', 'Ship Date']:
#         if col in df_display.columns:
#             df_display[col] = pd.to_datetime(df_display[col]).dt.strftime('%m/%d/%Y')
    
#     # Get column names for the data table, excluding some utility columns
#     columns_to_display = [col for col in df_display.columns if col in ['Customer Name', 'City', 'Category', 'Order Date', 'Segment', 'Ship Date', 'Ship Mode', 'State', 'Sub-Category', 'Revenue']]
    
#     return dbc.Container([
#         # Header
#         dbc.Row([
#             dbc.Col([
#                 html.H1("Records", className="display-5 fw-bold"),
#                 html.P("Browse and filter detailed transaction records."),
#             ], width=12),
#         ], className="py-3"),
        
#         # Data table
#         dbc.Row([
#             dbc.Col([
#                 dbc.Card(
#                     dbc.CardBody([
#                         html.H5(f"Order Records ({len(df_display)} rows)", className="card-title mb-3"),
#                         dash_table.DataTable(
#                             id='records-table',
#                             columns=[{"name": i, "id": i} for i in columns_to_display],
#                             data=df_display[columns_to_display].to_dict('records'),
#                             page_size=15,
#                             filter_action="native",
#                             sort_action="native",
#                             style_table={'overflowX': 'auto', 'height': '600px', 'overflowY': 'auto'},
#                             style_cell={
#                                 'overflow': 'hidden',
#                                 'textOverflow': 'ellipsis',
#                                 'maxWidth': 0,
#                                 'fontSize': '12px',
#                                 'fontFamily': 'Arial',
#                                 'padding': '5px'
#                             },
#                             style_header={
#                                 'backgroundColor': 'rgb(240, 240, 240)',
#                                 'fontWeight': 'bold',
#                                 'border': '1px solid #ddd'
#                             },
#                             style_data={
#                                 'whiteSpace': 'normal',
#                                 'height': 'auto',
#                                 'lineHeight': '15px',
#                                 'border': '1px solid #ddd'
#                             },
#                             style_data_conditional=[
#                                 {
#                                     'if': {'row_index': 'odd'},
#                                     'backgroundColor': 'rgb(248, 248, 248)'
#                                 }
#                             ],
#                         )
#                     ]),
#                     className="shadow-sm"
#                 )
#             ], width=12)
#         ])
#     ], fluid=True)

def generate_records_tab(selected_year):
    """Generate the Records tab content with a styled card layout"""

    df_year = df[df["Year"] == int(selected_year)]

    # Format display columns
    df_display = df_year.copy()

    for col in ['Revenue', 'Profit']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

    for col in ['Order Date', 'Ship Date']:
        if col in df_display.columns:
            df_display[col] = pd.to_datetime(df_display[col])

    # Calculate shipping days
    df_display['Shipping Days'] = (df_display['Ship Date'] - df_display['Order Date']).dt.days

    # Convert dates back to strings for display
    df_display['Order Date'] = df_display['Order Date'].dt.strftime('%m/%d/%Y')
    df_display['Ship Date'] = df_display['Ship Date'].dt.strftime('%m/%d/%Y')

    # create Column headers row
    header_row = dbc.Row([
        dbc.Col(html.Div("Customer Name", className="fw-bold text-muted small"), width=2),
        dbc.Col(html.Div("State & City", className="fw-bold text-muted small"), width=2),
        dbc.Col(html.Div("Category", className="fw-bold text-muted small"), width=2),
        dbc.Col(html.Div("Shipping Days", className="fw-bold text-muted small"), width=2),
        dbc.Col(html.Div("Segment & Ship Mode", className="fw-bold text-muted small"), width=2),
        dbc.Col(html.Div("Order Details", className="fw-bold text-muted small text-end"), width=2)
    ], className="mb-2")

    cards = []
    for _, row in df_display.iterrows():
        card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H6(row.get('Customer Name', ''), className="fw-bold mb-1"),
                        html.Div(f"{row.get('Customer ID', '')}", className="text-muted small"),
                        html.Div(f"{row.get('Order Date', '')}", className="text-muted small")
                    ]), width=2),
                    dbc.Col(html.Div([
                        html.H6(row.get('State', ''), className="fw-bold mb-1"),
                        html.Div(row.get('City', ''), className="text-muted small")
                    ]), width=2),
                    dbc.Col(html.Div([
                        html.Div(row.get('Category', '')),
                        html.Div(row.get('Sub-Category', ''), className="text-muted small")
                    ]), width=2),
                    dbc.Col(html.Div([
                        html.Div(f"{row.get('Shipping Days', '')} days", className="text-muted small")
                    ]), width=2),
                    dbc.Col(html.Div([
                        html.Div(row.get('Segment', '')),
                        html.Div(row.get('Ship Mode', ''), className="text-muted small")
                    ]), width=2),
                    dbc.Col(html.Div([
                        html.Div([
                            html.Span("Revenue:", className="fw-bold me-1"),
                            html.Span(row.get('Revenue', ''))
                            #light border
                        ], className="bg-light border rounded px-3 py-1 fw-bold text-dark d-inline-block")
                    ]), width=2, className="text-end")
                ])
            ])
        ], className="mb-2 shadow-sm")

        cards.append(card)

    return dbc.Container([

        dbc.Row([
            dbc.Col([
                html.H1("Records", className="display-5 fw-bold"),
                html.P("Access a comprehensive summary of KPIs and vital insights.")
            ])
        ], className="py-3"),

        dbc.Row([
            dbc.Col(html.H5(f"Order Records ({len(df_display)} rows)", className="mb-3 fw-semibold"))
        ]),

        header_row,  # Inserted header row here

        dbc.Row([
            dbc.Col(cards, width=12)
        ])

    ], fluid=True)


# Function to generate sales dashboard content
def generate_sales_dashboard(selected_year, selected_metric):
    """Generate the Sales Dashboard content"""
    df_filter = df[df["Year"] == int(selected_year)]
    df_previous_yr = df[df["Year"] == int(selected_year) - 1]
    
    # Calculate metrics
    total_customers = df_filter["Customer ID"].nunique()
    total_revenue = f"${df_filter['Revenue'].sum()/1000:,.0f}K"
    total_quantity = f"{df_filter['Quantity'].sum():,.0f}"
    total_profit = f"${df_filter['Profit'].sum()/1000:,.0f}K"
    total_states = df_filter["State"].nunique()
    
    # Calculate monthly data for sparklines
    monthly_customers = df_filter.resample("M", on = 'Order Date')["Customer ID"].nunique().values
    monthly_revenue = df_filter.resample("M", on = 'Order Date')["Revenue"].sum().values
    monthly_quantity = df_filter.resample("M", on = 'Order Date')["Quantity"].sum().values
    monthly_profit = df_filter.resample("M", on = 'Order Date')["Profit"].sum().values
    prev_monthly_customers= df_previous_yr.resample("M", on = 'Order Date')["Customer ID"].nunique().values
    prev_monthly_revenue = df_previous_yr.resample("M", on = 'Order Date')["Revenue"].sum().values
    prev_monthly_quantity = df_previous_yr.resample("M", on = 'Order Date')["Quantity"].sum().values
    prev_monthly_profit = df_previous_yr.resample("M", on = 'Order Date')["Profit"].sum().values
    
    # Calculate percentage changes
    revenue_change = calc_percentage_change(df_filter["Revenue"].sum(), df_previous_yr["Revenue"].sum())
    quantity_change = calc_percentage_change(df_filter["Quantity"].sum(), df_previous_yr["Quantity"].sum())
    customers_change = calc_percentage_change(df_filter["Customer ID"].nunique(), df_previous_yr["Customer ID"].nunique())
    profit_change = calc_percentage_change(df_filter["Profit"].sum(), df_previous_yr["Profit"].sum())
    
    # Handle different metric types
    if selected_metric in ["Revenue", "Profit", "Quantity"]:
        grouped_state_df = df_filter.groupby('State', as_index = False)[selected_metric].sum()
        grouped_state_py = df_previous_yr.groupby('State', as_index=False)[selected_metric].sum()
    elif selected_metric=="Customers":
        grouped_state_df = df_filter.groupby('State', as_index = False)["Customer ID"].nunique()
        grouped_state_py = df_previous_yr.groupby('State', as_index=False)["Customer ID"].nunique()
        selected_metric = "Customer ID"
    
    # Merge state data
    min_max_df = pd.merge(grouped_state_df[['State', selected_metric]], grouped_state_py[['State', selected_metric]], on = 'State', how = 'outer', suffixes = ('_current', '_previous')).fillna(0)
    
    # Find best and worst performing states
    best_state = min_max_df.loc[min_max_df[f'{selected_metric}_current'].idxmax()]
    worst_state = min_max_df.loc[min_max_df[f'{selected_metric}_current'].idxmin()]
    max_state = best_state['State']
    max_val = best_state[f'{selected_metric}_current']
    min_state = worst_state['State']
    min_val = worst_state[f'{selected_metric}_current']


    
    
    # Create figures
    state_map_fig = create_state_map(grouped_state_df, grouped_state_py, selected_year, selected_metric)

    if selected_metric == "Customer ID":
        fur_cat = df_filter[df_filter["Category"] == 'Furniture'][selected_metric].nunique()
        sup_cat = df_filter[df_filter["Category"] == 'Office Supplies'][selected_metric].nunique()
        tech_cat = df_filter[df_filter["Category"] == 'Technology'][selected_metric].nunique()
        fur_cat_prev = df_previous_yr[df_previous_yr["Category"] == 'Furniture'][selected_metric].nunique()
        sup_cat_prev = df_previous_yr[df_previous_yr["Category"] == 'Office Supplies'][selected_metric].nunique()
        tech_cat_prev = df_previous_yr[df_previous_yr["Category"] == 'Technology'][selected_metric].nunique()
    else:
        fur_cat = df_filter[df_filter["Category"] == 'Furniture'][selected_metric].sum()
        sup_cat = df_filter[df_filter["Category"] == 'Office Supplies'][selected_metric].sum()
        tech_cat = df_filter[df_filter["Category"] == 'Technology'][selected_metric].sum()
        fur_cat_prev = df_previous_yr[df_previous_yr["Category"] == 'Furniture'][selected_metric].sum()
        sup_cat_prev = df_previous_yr[df_previous_yr["Category"] == 'Office Supplies'][selected_metric].sum()
        tech_cat_prev = df_previous_yr[df_previous_yr["Category"] == 'Technology'][selected_metric].sum()

    category_df = pd.DataFrame({"Category":["Furniture", "Office Supplies", "Technology"],
                                "value": [fur_cat, sup_cat, tech_cat],
                                "previous value": [fur_cat_prev, sup_cat_prev, tech_cat_prev]})
    
    category_df["% Change"] = ((category_df["value"] - category_df["previous value"]) / category_df["previous value"]) * 100
    category_fig = create_horizontal_bar(category_df, "Category", selected_metric)
    #region_fig = create_horizontal_bar(None, "Region", is_category=False)
    
    # Calculate national average
    if selected_metric == "Customer ID":    
        national_avg = (total_customers / total_states)
    else:    
        national_avg = ((df_filter[selected_metric].sum() / total_states).round(0))
    
    # Return the dashboard content
    return dbc.Container([
        dbc.Row([
            # LEFT: 4 KPI cards in a 2x2 grid
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        # Add fixed height to the card
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Customers", className="card-title"),
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                            html.H3(
                                                f"{total_customers:,.0f}",
                                                className="card-text", style={"color": "#1c4e8b"}),
                                            html.Span(
                                                f"{customers_change:.2f}% ",
                                                style={"color": "green" if customers_change > 0 else "red", "fontSize": "14px"},
                                            ),
                                            html.Span(
                                                "vs PY",
                                                style={"color":"grey"}
                                            ),
                                        ]),
                                        width=6
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            figure=make_sparkline(monthly_customers, prev_monthly_customers, selected_year, int(selected_year)-1, "Customers"),
                                            config={"displayModeBar": False},
                                            style={"height": "100px"}
                                        ),
                                        width=6
                                    )
                                ]) 
                            ]),
                            className="shadow-sm h-100"  # Add h-100 class for full height
                        ), 
                        width=6,
                        # Add className to adjust column height
                        className="d-flex" 
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Revenue", className="card-title"),
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                            html.H3(
                                                total_revenue, className="card-text",
                                                style={"color": "#1c4e8b"}),
                                            html.Span(
                                                f"{revenue_change:.2f}% ",
                                                style={"color": "green" if revenue_change > 0 else "red", "fontSize": "14px"},
                                            ),
                                            html.Span(
                                                "vs PY",
                                                style={"color":"grey"}
                                            ),
                                        ]),
                                        width=6
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            figure=make_sparkline(monthly_revenue, prev_monthly_revenue, selected_year, int(selected_year)-1, "Revenue"),
                                            config={"displayModeBar": False},
                                            style={"height": "100px"}
                                        ),
                                        width=6
                                    )
                                ])    
                            ]),
                            className="shadow-sm h-100"  # Add h-100 class for full height
                        ), 
                        width=6,
                        className="d-flex"  # Add d-flex for consistent height
                    ),
                ], className="mb-3 h-50"),  # Add h-50 to make each row exactly 50% of total height
                dbc.Row([
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Quantity", className="card-title"),
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                            html.H3(total_quantity, className="card-text",
                                                style={"color": "#1c4e8b"}),
                                            html.Span(
                                                f"{quantity_change:.2f}% ",
                                                style={"color": "green" if quantity_change > 0 else "red", "fontSize": "14px"},
                                            ),
                                            html.Span(
                                                "vs PY",
                                                style={"color":"grey"}
                                            ),
                                        ]),
                                        width=6
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            figure=make_sparkline(monthly_quantity, prev_monthly_quantity, selected_year, int(selected_year)-1, "Quantity"),
                                            config={"displayModeBar": False},
                                            style={"height": "100px"}
                                        ),
                                        width=6
                                    )
                                ])
                            ]),
                            className="shadow-sm h-100"  # Add h-100 class for full height
                        ), 
                        width=6,
                        className="d-flex"  # Add d-flex for consistent height
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Profit", className="card-title"),
                                dbc.Row([
                                    dbc.Col(
                                        html.Div([
                                            html.H3(total_profit, className="card-text",
                                                style={"color": "#1c4e8b"}),
                                            html.Span(
                                                f"{profit_change:.2f}% ",
                                                style={"color": "green" if profit_change > 0 else "red", "fontSize": "14px"},
                                            ),
                                            html.Span(
                                                "vs PY",
                                                style={"color":"grey"}
                                            ),
                                        ]),
                                        width=6
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            figure=make_sparkline(monthly_profit, prev_monthly_profit, selected_year, int(selected_year)-1, "Profit"),
                                            config={"displayModeBar": False},
                                            style={"height": "100px"}
                                        ),
                                        width=6
                                    )
                                ])
                            ]),
                            className="shadow-sm h-100"  # Add h-100 class for full height
                        ), 
                        width=6,
                        className="d-flex"  # Add d-flex for consistent height
                    ),
                ], className="mb-3 h-50"),  # Add h-50 to make each row exactly 50% of total height
            ], width=6, className="d-flex flex-column", style={"height": "500px"}),  # Set fixed height for the entire column
            
            # RIGHT: Interactive bar chart
            dbc.Col([
                dcc.Graph(
                    id="bar-chart",
                    figure=create_bar_chart(metric=selected_metric, selected_year=selected_year),
                    style={"height": "500px"}  # Match this height with the KPI cards container
                )
            ], width=6),
        ], className="g-3 mb-3"),
        
        # US State Map section
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H5("State", className="card-title"),
                        html.P("Click a state to filter the view", className="text-muted"),
                        dbc.Row([
                            dbc.Col([
                                html.H5("National Average", className="mt-3"),
                                html.H3(
                                    f"${national_avg:,.0f}", 
                                    className="card-text",
                                    style={"color": "#1c4e8b", "fontSize": "25px"}
                                ),
                                html.Div([
                                    html.P("Best performing:", className="mb-0 mt-3"),
                                    html.P(f"{max_state} (${max_val:,.0f})", 
                                        style={"fontWeight": "bold"}),
                                    html.P("Worst performing:", className="mb-0 mt-2"),
                                    html.P(f"{min_state} (${min_val:,.0f})", 
                                        style={"fontWeight": "bold"}),
                                ])
                            ], width=3),
                            dbc.Col([
                                dcc.Graph(
                                    figure=state_map_fig,
                                    config={"displayModeBar": False},
                                    style={"height": "300px"}
                                )
                            ], width=9)
                        ])
                    ]),
                    className="shadow-sm"
                )
            ], width=6),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    
                            category_fig
                            
                        
                                ], style={
                                   "display": "flex",
                                   "justifyContent": "center"})
                            ], width = 6),
                             dbc.Col([
                                html.Div([
                                    category_fig
                ], style={
                                   "display": "flex",
                                   "justifyContent": "center"})
            ], width=6)
                        ], style={"displayflex":"flex", "height":"100%"})
                        
                    ]),
                    className="shadow-sm"
                )
            ]),
           
            
        ], className="g-3 mb-3"),
    ])
        # Category and Region cards side by side
    #     dbc.Row([
    #         # Left: Category horizontal bar chart
    #         dbc.Col([
    #             dbc.Card(
    #                 dbc.CardBody([
    #                     dcc.Graph(
    #                         figure=category_fig,
    #                         config={"displayModeBar": False},
    #                         style={"height": "150px"}
    #                     )
    #                 ]),
    #                 className="shadow-sm"
    #             )
    #         ], width=4),
            
    #         # Right: Region horizontal bar chart
    #         dbc.Col([
    #             dbc.Card(
    #                 dbc.CardBody([
    #                     dcc.Graph(
    #                         figure=region_fig,
    #                         config={"displayModeBar": False},
    #                         style={"height": "150px"}
    #                     )
    #                 ]), 
    #                 className="shadow-sm"
    #             )
    #         ], width=4)
    #     ], className="g-3 mb-3"),
    # ], fluid=True)

# ------------------------------------------------------------------
# 8. App Layout - UPDATED
# ------------------------------------------------------------------
app.layout = html.Div([
    dbc.Card(
        dbc.Container([
            top_nav,
            header,
            # Move tabs to the main layout
            # dcc.Tabs(
            #     id="tabs",
            #     value="sales_dashboard",  # Default tab selected
            #     children=[
            #         dcc.Tab(
            #             label="Sales Dashboard",
            #             value="sales_dashboard",
            #             children=[html.Div(id="sales-dashboard-content")]  # This div will be populated by the callback
            #         ),
            #         dcc.Tab(
            #             label="Insights",
            #             value="insights",
            #             children=[html.Div(id="insights-content")]  # This div will be populated by the callback
            #         ),
            #         dcc.Tab(
            #             label="Records",
            #             value="records",
            #             children=[html.Div(id="records-content")]  # This div will be populated by the callback
            #         )
            #     ],
            #     style={"backgroundColor": "#fff", "padding": "10px"}
            # ),
            dcc.Store(id="selected-tab", data="sales_dashboard"),
            html.Div(id="tab-content")  # This is a placeholder that will be populated by the callback
        ]),
    )
])

# ------------------------------------------------------------------
# 9. Updated Callbacks for Tab Navigation
# ------------------------------------------------------------------
# Create a callback to handle tab selection
@callback(
    Output("tab-content", "children"),
    [
        Input("selected-tab", "data"),
        Input("year-dropdown", "value"),
        Input("metric-dropdown", "value")
    ]
)
def render_tab_content(selected_tab, selected_year, selected_metric):
    """
    This callback renders the content for the selected tab
    """
    if selected_tab == "sales_dashboard" or selected_tab is None:  # Default to sales dashboard if no tab selected
        return generate_sales_dashboard(selected_year, selected_metric)
    elif selected_tab == "insights":
        return create_insights_tab(df, selected_year)
    elif selected_tab == "records":
        return generate_records_tab(selected_year)
    
    # Default fallback
    return html.Div("Please select a tab")

@callback(
    Output("selected-tabs", "data"),
    [
        Input("nav-dashboard", "n_clicks"),
        Input("nav-insight", "n_clicks"),
        Input("nav-records", "n_clicks")
    ]    
)
def switch_tab(n1, n2, n3):
    trigger = ctx.triggered_id
    if trigger == "nav-dashboard":
        return "sales_dashboard"
    if trigger == "nav-insight":
        return "insights"
    if trigger == "nav-records":
        return "records"
    return dash.no_update


# ------------------------------------------------------------------
# 10. Run the app
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
