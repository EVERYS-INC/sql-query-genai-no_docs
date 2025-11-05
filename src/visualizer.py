import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class DataVisualizer:
    def __init__(self):
        self.color_scheme = px.colors.qualitative.Set3
    
    def create_chart(self, df, chart_type, title=""):
        try:
            if chart_type == "line":
                return self._create_line_chart(df, title)
            elif chart_type == "bar":
                return self._create_bar_chart(df, title)
            elif chart_type == "scatter":
                return self._create_scatter_chart(df, title)
            elif chart_type == "pie":
                return self._create_pie_chart(df, title)
            elif chart_type == "heatmap":
                return self._create_heatmap(df, title)
            else:
                return None
                
        except Exception as e:
            print(f"グラフ作成エラー: {str(e)}")
            return None
    
    def _create_line_chart(self, df, title):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        if len(numeric_cols) == 0:
            return None
        
        if len(non_numeric_cols) > 0:
            x_col = non_numeric_cols[0]
        else:
            x_col = df.index.name if df.index.name else 'index'
            df = df.reset_index()
        
        fig = go.Figure()
        
        for i, col in enumerate(numeric_cols):
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(color=self.color_scheme[i % len(self.color_scheme)])
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="値",
            hovermode='x unified',
            template="plotly_white"
        )
        
        return fig
    
    def _create_bar_chart(self, df, title):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        if len(numeric_cols) == 0:
            return None
        
        if len(non_numeric_cols) > 0:
            x_col = non_numeric_cols[0]
            y_col = numeric_cols[0]
            
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col,
                title=title,
                color_discrete_sequence=self.color_scheme
            )
        else:
            fig = px.bar(
                df,
                y=numeric_cols[0],
                title=title,
                color_discrete_sequence=self.color_scheme
            )
        
        fig.update_layout(template="plotly_white")
        return fig
    
    def _create_scatter_chart(self, df, title):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return self._create_bar_chart(df, title)
        
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=title,
            color_discrete_sequence=self.color_scheme
        )
        
        if len(numeric_cols) > 2:
            fig.update_traces(
                marker=dict(
                    size=df[numeric_cols[2]],
                    sizemode='area',
                    sizeref=2.*max(df[numeric_cols[2]])/(40.**2),
                    sizemin=4
                )
            )
        
        fig.update_layout(template="plotly_white")
        return fig
    
    def _create_pie_chart(self, df, title):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        if len(numeric_cols) == 0:
            return None
        
        if len(non_numeric_cols) > 0:
            labels_col = non_numeric_cols[0]
            values_col = numeric_cols[0]
            
            fig = px.pie(
                df,
                names=labels_col,
                values=values_col,
                title=title,
                color_discrete_sequence=self.color_scheme
            )
        else:
            fig = px.pie(
                values=df[numeric_cols[0]],
                names=df.index,
                title=title,
                color_discrete_sequence=self.color_scheme
            )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template="plotly_white")
        return fig
    
    def _create_heatmap(self, df, title):
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=numeric_df.values,
            x=numeric_df.columns,
            y=df.index if df.index.name else list(range(len(df))),
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        
        return fig