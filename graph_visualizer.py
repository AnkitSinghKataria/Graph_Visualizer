#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================
# Full GraphVisualizer Library
# Supports: Univariate, Bivariate, Multivariate, Datetime Analysis
# Backends: Seaborn, Matplotlib, Plotly
# ==========================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
from mpl_toolkits.mplot3d import Axes3D

class GraphVisualizer:
    def __init__(self, df=None):
        self.df = df
        self.backend = None
        self.dashboard = []
        self.exit_flag = False

    def run(self):
        print("\n📊 Welcome to GraphVisualizer!")

        # Step 1: Load DataFrame if not provided
        if self.df is None:
            print("\nChoose your data source:")
            print("1. Load from Seaborn sample dataset")
            print("2. Use your own DataFrame (assign to self.df before run)")
            df_choice = input("Enter choice (1 or 2): ").strip()
            if df_choice == "1":
                print("\nSeaborn Datasets:")
                print(", ".join(sns.get_dataset_names()[:10]), "...")
                name = input("Enter dataset name: ").strip()
                try:
                    self.df = sns.load_dataset(name)
                    print(f"✅ Loaded dataset: {name}")
                except Exception:
                    print("❌ Dataset not found or load error.")
                    return
            else:
                print("⚠️ Please assign your own DataFrame to `self.df` before proceeding.")
                return

        # Step 2: Choose backend
        print("\nChoose visualization backend:")
        print("1. Seaborn")
        print("2. Matplotlib")
        print("3. Plotly")
        bmap = {"1": "seaborn", "2": "matplotlib", "3": "plotly"}
        self.backend = bmap.get(input("Enter choice: ").strip(), "seaborn")
        print(f"📌 Using {self.backend} backend.")

        # Main loop
        self.main_menu()

    def main_menu(self):
        while not self.exit_flag:
            print("\nMain Menu")
            print("1. Univariate Analysis")
            print("2. Bivariate Analysis")
            print("3. Multivariate Analysis")
            print("4. Datetime Analysis")
            print("5. Export Dashboard")
            print("6. Exit")
            choice = input("Enter choice: ").strip()
            if choice == "1":
                self.univariate_menu()
            elif choice == "2":
                self.bivariate_menu()
            elif choice == "3":
                self.multivariate_menu()
            elif choice == "4":
                self.datetime_menu()
            elif choice == "5":
                self.export_dashboard()
            elif choice == "6":
                print("👋 Exiting GraphVisualizer.")
                self.exit_flag = True
            else:
                print("❌ Invalid input.")

    def choose_from_list(self, prompt, options):
        """
        Displays a numbered list of options and returns the chosen value.
        """
        print(f"\n{prompt}")
        for i, opt in enumerate(options):
            print(f"{i+1}. {opt}")
        idx = int(input("Choose number: ").strip()) - 1
        return options[idx]

    def export_dashboard(self):
        """
        Saves all figures in the dashboard list to 'eda_outputs/' folder.
        Handles both Matplotlib and Plotly figures.
        """
        if not self.dashboard:
            print("⚠️ No dashboard plots yet.")
            return

        os.makedirs("eda_outputs", exist_ok=True)
        for i, (fig, desc, ftype) in enumerate(self.dashboard):
            filename = desc.replace(" ", "_")
            path = f"eda_outputs/{filename}_{i}.{ftype}"

            # If it's a Matplotlib figure, use savefig
            if hasattr(fig, "savefig"):
                try:
                    fig.savefig(path)
                except Exception as e:
                    print(f"❌ Failed to save Matplotlib figure '{desc}': {e}")

            # If it's a Plotly figure, use write_image
            elif hasattr(fig, "write_image"):
                try:
                    fig.write_image(path)
                except Exception as e:
                    print(f"❌ Failed to save Plotly figure '{desc}': {e}")

            else:
                print(f"⚠️ Unknown figure type for '{desc}', cannot export.")

        print("✅ Dashboard exported to 'eda_outputs/'.")


    # -------------------------
    # Univariate Analysis
    # -------------------------
    def univariate_menu(self):
        print("\n--- Univariate Analysis ---")
        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        cat_cols = self.df.select_dtypes(include='object').columns.tolist()
        print("1. Numerical Columns")
        print("2. Categorical Columns")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            col = self.choose_from_list("Select numerical column:", num_cols)
            print("\nChoose graph:")
            plots = ["Histogram", "Boxplot", "Violinplot", "KDE", "Rugplot", "ECDF", "Stripplot", "Swarmplot", "Scatter (from Plotly)", "Jointplot"]
            plot_type = self.choose_from_list("Select plot type:", plots)
            hue = None
            if cat_cols:
                add_hue = input("Add hue/color? (y/n): ").strip().lower()
                if add_hue == "y":
                    hue = self.choose_from_list("Select hue column:", cat_cols)
            self.plot_univariate_numeric(col, plot_type, hue)

        elif choice == "2":
            col = self.choose_from_list("Select categorical column:", cat_cols)
            print("\nChoose graph:")
            plots = ["Countplot", "Barplot", "Pie Chart"]
            plot_type = self.choose_from_list("Select plot type:", plots)
            hue = None
            if cat_cols:
                add_hue = input("Add hue/color? (y/n): ").strip().lower()
                if add_hue == "y":
                    hue = self.choose_from_list("Select hue column:", cat_cols)
            self.plot_univariate_categorical(col, plot_type, hue)

        else:
            return

    def plot_univariate_numeric(self, col, plot_type, hue=None):
        title = f"{plot_type} of {col}"
        fig = None
        ftype = "png"

        if self.backend == "seaborn":
            plt.figure(figsize=(8, 4))
            if plot_type == "Histogram":
                sns.histplot(data=self.df, x=col, hue=hue, bins=30, kde=False)
            elif plot_type == "Boxplot":
                if hue:
                    sns.boxplot(data=self.df, x=hue, y=col)
                else:
                    sns.boxplot(data=self.df, y=col)
            elif plot_type == "Violinplot":
                if hue:
                    sns.violinplot(data=self.df, x=hue, y=col)
                else:
                    sns.violinplot(data=self.df, y=col)
            elif plot_type == "KDE":
                sns.kdeplot(data=self.df, x=col, hue=hue)
            elif plot_type == "Rugplot":
                sns.rugplot(data=self.df, x=col, hue=hue)
            elif plot_type == "ECDF":
                sns.ecdfplot(data=self.df, x=col, hue=hue)
            elif plot_type == "Stripplot":
                if hue:
                    sns.stripplot(data=self.df, x=hue, y=col, dodge=True)
                else:
                    sns.stripplot(data=self.df, y=col)
            elif plot_type == "Swarmplot":
                if hue:
                    sns.swarmplot(data=self.df, x=hue, y=col, dodge=True)
                else:
                    sns.swarmplot(data=self.df, y=col)
            elif plot_type == "Scatter (from Plotly)":
                print("⚠️ Scatter selected: switching to Plotly for interactivity.")
                self.backend = "plotly"
                self.plot_univariate_numeric(col, "Histogram", hue)
                return
            elif plot_type == "Jointplot":
                if hue:
                    sns.jointplot(data=self.df, x=col, y=hue, kind="scatter")
                else:
                    sns.jointplot(data=self.df, x=col, y=col, kind="kde")
                plt.tight_layout()
                plt.show()
                fig = plt.gcf()
                self.dashboard.append((fig, title, ftype))
                return
            else:
                print("⚠️ Unsupported plot type for Seaborn.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "matplotlib":
            plt.figure(figsize=(8, 4))
            if plot_type == "Histogram":
                plt.hist(self.df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
                plt.xlabel(col)
                plt.ylabel("Count")
            elif plot_type == "Boxplot":
                plt.boxplot(self.df[col].dropna(), patch_artist=True)
                plt.ylabel(col)
            elif plot_type == "Violinplot":
                plt.violinplot(self.df[col].dropna())
                plt.ylabel(col)
            elif plot_type == "KDE":
                data = self.df[col].dropna()
                sns.kdeplot(data, fill=True, color='steelblue')
                plt.xlabel(col)
                plt.ylabel("Density")
            elif plot_type == "Rugplot":
                data = self.df[col].dropna()
                for val in data:
                    plt.axvline(val, color='black', linewidth=0.5)
                plt.xlabel(col)
                plt.yticks([])
            elif plot_type == "ECDF":
                data = self.df[col].dropna().sort_values()
                y = (data.rank(method="first") - 1) / (len(data) - 1)
                plt.plot(data, y, marker='.', linestyle='none')
                plt.xlabel(col)
                plt.ylabel("ECDF")
            elif plot_type == "Stripplot":
                data = self.df[col].dropna()
                jitter = (np.random.rand(len(data)) - 0.5) * 0.1
                plt.scatter(data, jitter)
                plt.xlabel(col)
                plt.yticks([])
            elif plot_type == "Swarmplot":
                print("⚠️ Swarmplot not directly supported in Matplotlib. Using Stripplot fallback.")
                data = self.df[col].dropna()
                jitter = (np.random.rand(len(data)) - 0.5) * 0.1
                plt.scatter(data, jitter)
                plt.xlabel(col)
                plt.yticks([])
            elif plot_type == "Jointplot":
                xdat = self.df[col].dropna()
                ydat = xdat
                plt.hist2d(xdat, ydat, bins=30, cmap='Blues')
                plt.xlabel(col)
                plt.ylabel(col)
                plt.colorbar(label='Counts')
            else:
                print("⚠️ Unsupported plot type for Matplotlib.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "plotly":
            fig = None
            if plot_type == "Histogram":
                fig = px.histogram(self.df, x=col, color=hue, title=title)
            elif plot_type == "Boxplot":
                fig = px.box(self.df, y=col, color=hue, title=title)
            elif plot_type == "Violinplot":
                fig = px.violin(self.df, y=col, color=hue, title=title, box=True, points="all")
            elif plot_type == "KDE":
                fig = px.density_contour(self.df, x=col, color=hue, title=title)
            elif plot_type == "ECDF":
                fig = px.ecdf(self.df, x=col, color=hue, title=title)
            elif plot_type == "Stripplot":
                fig = px.strip(self.df, y=col, color=hue, title=title)
            elif plot_type == "Swarmplot":
                fig = px.strip(self.df, y=col, color=hue, title=f"Swarm-like: {title}")
            elif plot_type == "Scatter (from Plotly)":
                fig = px.scatter(self.df, x=col, y=col, color=hue, title=title)
            elif plot_type == "Jointplot":
                fig = px.density_heatmap(self.df, x=col, y=col, marginal_x="histogram", marginal_y="histogram", title=title)
            else:
                print("⚠️ Unsupported plot type for Plotly.")
                return
            fig.show()

        # Save to dashboard if fig is defined
        if fig:
            self.dashboard.append((fig, title, ftype))

    def plot_univariate_categorical(self, col, plot_type, hue=None):
        title = f"{plot_type} of {col}"
        fig = None
        ftype = "png"

        if self.backend == "seaborn":
            plt.figure(figsize=(8, 4))
            if plot_type == "Countplot":
                sns.countplot(data=self.df, x=col, hue=hue)
            elif plot_type == "Barplot":
                counts = self.df[col].value_counts().reset_index()
                counts.columns = [col, "count"]
                sns.barplot(data=counts, x=col, y="count", hue=hue if hue else None)
            elif plot_type == "Pie Chart":
                pie_data = self.df[col].value_counts()
                plt.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
                plt.axis("equal")
            else:
                print("⚠️ Unsupported categorical plot for Seaborn.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "matplotlib":
            counts = self.df[col].value_counts()
            plt.figure(figsize=(8, 4))
            if plot_type in ["Countplot", "Barplot"]:
                plt.bar(counts.index.astype(str), counts.values, color='coral', edgecolor='black')
                plt.xticks(rotation=45)
            elif plot_type == "Pie Chart":
                plt.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
                plt.axis("equal")
            else:
                print("⚠️ Unsupported categorical plot for Matplotlib.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "plotly":
            if plot_type == "Countplot":
                fig = px.histogram(self.df, x=col, color=hue, title=title)
            elif plot_type == "Barplot":
                bar_data = self.df[col].value_counts().reset_index()
                bar_data.columns = [col, "count"]
                fig = px.bar(bar_data, x=col, y="count", color=hue, title=title)
            elif plot_type == "Pie Chart":
                pie_data = self.df[col].value_counts().reset_index()
                pie_data.columns = [col, "count"]
                fig = px.pie(pie_data, values="count", names=col, title=title)
            else:
                print("⚠️ Unsupported categorical plot for Plotly.")
                return
            fig.show()

        if fig:
            self.dashboard.append((fig, title, ftype))

    # -------------------------
    # Bivariate Analysis
    # -------------------------
    def bivariate_menu(self):
        print("\n--- Bivariate Analysis ---")
        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        cat_cols = self.df.select_dtypes(include='object').columns.tolist()

        print("1. Numerical vs Numerical")
        print("2. Numerical vs Categorical")
        print("3. Categorical vs Categorical")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            self.bivariate_numeric_numeric(num_cols, cat_cols)
        elif choice == "2":
            self.bivariate_numeric_categorical(num_cols, cat_cols)
        elif choice == "3":
            self.bivariate_categorical_categorical(cat_cols)
        else:
            return

    def bivariate_numeric_numeric(self, num_cols, cat_cols):
        # Select x, y, plot type, optional hue
        x = self.choose_from_list("Select X-axis (numerical):", num_cols)
        y = self.choose_from_list("Select Y-axis (numerical):", num_cols)
        print("\nChoose plot type:")
        plots = ["Scatter", "Line", "Hexbin", "Jointplot", "Regression (Seaborn)", "Heatmap (2D histogram)"]
        plot_type = self.choose_from_list("Select type:", plots)

        hue = None
        if cat_cols and plot_type in ["Scatter", "Line", "Jointplot", "Regression (Seaborn)"]:
            add_hue = input("Add hue/color? (y/n): ").strip().lower()
            if add_hue == "y":
                hue = self.choose_from_list("Select hue column:", cat_cols)

        title = f"{plot_type} of {x} vs {y}"
        fig = None
        ftype = "png"

        if self.backend == "seaborn":
            plt.figure(figsize=(8, 5))
            if plot_type == "Scatter":
                sns.scatterplot(data=self.df, x=x, y=y, hue=hue)
            elif plot_type == "Line":
                sns.lineplot(data=self.df, x=x, y=y, hue=hue)
            elif plot_type == "Hexbin":
                plt.hexbin(self.df[x], self.df[y], gridsize=30, cmap="Blues")
                plt.colorbar(label="counts")
            elif plot_type == "Jointplot":
                if hue:
                    sns.jointplot(data=self.df, x=x, y=y, hue=hue)
                else:
                    sns.jointplot(data=self.df, x=x, y=y, kind="scatter")
            elif plot_type == "Regression (Seaborn)":
                sns.lmplot(data=self.df, x=x, y=y, hue=hue, line_kws={"color":"red"})
            elif plot_type == "Heatmap (2D histogram)":
                sns.histplot(data=self.df, x=x, y=y, bins=30, pthresh=.1, cmap="mako")
            else:
                print("⚠️ Unsupported plot type for Seaborn.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "matplotlib":
            plt.figure(figsize=(8, 5))
            if plot_type == "Scatter":
                plt.scatter(self.df[x], self.df[y], c='skyblue', edgecolor='black')
            elif plot_type == "Line":
                sorted_df = self.df.sort_values(by=x)
                plt.plot(sorted_df[x], sorted_df[y], color='blue')
            elif plot_type == "Hexbin":
                plt.hexbin(self.df[x], self.df[y], gridsize=30, cmap='Blues')
                plt.colorbar(label="counts")
            elif plot_type == "Jointplot":
                plt.hist2d(self.df[x], self.df[y], bins=30, cmap='Blues')
                plt.colorbar(label="counts")
            elif plot_type == "Regression (Seaborn)":
                from sklearn.linear_model import LinearRegression
                data = self.df[[x, y]].dropna()
                Xv = data[x].values.reshape(-1, 1)
                Yv = data[y].values
                model = LinearRegression().fit(Xv, Yv)
                preds = model.predict(Xv)
                plt.scatter(Xv, Yv, color='skyblue', edgecolor='black')
                plt.plot(Xv, preds, color='red')
            elif plot_type == "Heatmap (2D histogram)":
                plt.hist2d(self.df[x], self.df[y], bins=30, cmap='viridis')
                plt.colorbar(label="counts")
            else:
                print("⚠️ Unsupported plot type for Matplotlib.")
                return
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "plotly":
            if plot_type == "Scatter":
                fig = px.scatter(self.df, x=x, y=y, color=hue, title=title)
            elif plot_type == "Line":
                fig = px.line(self.df, x=x, y=y, color=hue, title=title)
            elif plot_type == "Hexbin":
                fig = px.density_heatmap(self.df, x=x, y=y, nbinsx=30, nbinsy=30, title=title)
            elif plot_type == "Jointplot":
                fig = px.scatter(self.df, x=x, y=y, marginal_x="histogram", marginal_y="histogram", color=hue, title=title)
            elif plot_type == "Regression (Seaborn)":
                fig = px.scatter(self.df, x=x, y=y, color=hue, trendline="ols", title=title)
            elif plot_type == "Heatmap (2D histogram)":
                fig = px.density_heatmap(self.df, x=x, y=y, nbinsx=30, nbinsy=30, title=title)
            else:
                print("⚠️ Unsupported plot type for Plotly.")
                return
            fig.show()

        # Save to dashboard
        if fig:
            self.dashboard.append((fig, title, ftype))

    def bivariate_numeric_categorical(self, num_cols, cat_cols):
        x = self.choose_from_list("Select numerical column (Y-axis):", num_cols)
        y = self.choose_from_list("Select categorical column (X-axis):", cat_cols)
        print("\nChoose plot type:")
        plots = ["Boxplot", "Violinplot", "Barplot", "Stripplot", "Swarmplot", "Pointplot"]
        plot_type = self.choose_from_list("Select type:", plots)
        hue = None
        if cat_cols:
            add_hue = input("Add hue? (y/n): ").strip().lower()
            if add_hue == "y":
                hue = self.choose_from_list("Select hue column:", cat_cols)

        title = f"{plot_type} of {x} vs {y}"
        fig = None
        ftype = "png"

        if self.backend == "seaborn":
            plt.figure(figsize=(8, 5))
            if plot_type == "Boxplot":
                sns.boxplot(data=self.df, x=y, y=x, hue=hue)
            elif plot_type == "Violinplot":
                sns.violinplot(data=self.df, x=y, y=x, hue=hue)
            elif plot_type == "Barplot":
                sns.barplot(data=self.df, x=y, y=x, hue=hue, estimator=lambda vals: vals.mean())
            elif plot_type == "Stripplot":
                sns.stripplot(data=self.df, x=y, y=x, hue=hue, dodge=True)
            elif plot_type == "Swarmplot":
                sns.swarmplot(data=self.df, x=y, y=x, hue=hue, dodge=True)
            elif plot_type == "Pointplot":
                sns.pointplot(data=self.df, x=y, y=x, hue=hue, dodge=0.5)
            else:
                print("⚠️ Unsupported plot type for Seaborn.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "matplotlib":
            plt.figure(figsize=(8, 5))
            data = self.df[[x, y]].dropna()
            if plot_type == "Boxplot":
                data.boxplot(column=x, by=y)
                plt.title(title)
                plt.suptitle("")
            elif plot_type == "Violinplot":
                sns.violinplot(data=data, x=y, y=x, hue=hue, dodge=True)
            elif plot_type == "Barplot":
                means = data.groupby(y)[x].mean()
                plt.bar(means.index.astype(str), means.values, color='lightgreen')
                plt.title(title)
            elif plot_type in ["Stripplot", "Swarmplot"]:
                sns.stripplot(data=data, x=y, y=x, hue=hue, dodge=True)
            elif plot_type == "Pointplot":
                grouped = data.groupby(y)[x].agg(['mean', 'std']).reset_index()
                plt.errorbar(grouped[y].astype(str), grouped['mean'], yerr=grouped['std'], fmt='o', ecolor='red', capsize=5)
                plt.title(title)
                plt.xlabel(y)
                plt.ylabel(x)
            else:
                print("⚠️ Unsupported plot type for Matplotlib.")
                return
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "plotly":
            if plot_type == "Boxplot":
                fig = px.box(self.df, x=y, y=x, color=hue, title=title)
            elif plot_type == "Violinplot":
                fig = px.violin(self.df, x=y, y=x, color=hue, box=True, title=title)
            elif plot_type == "Barplot":
                bar_data = self.df.groupby(y)[x].mean().reset_index()
                fig = px.bar(bar_data, x=y, y=x, color=hue, title=title)
            elif plot_type in ["Stripplot", "Swarmplot"]:
                fig = px.strip(self.df, x=y, y=x, color=hue, title=title, jitter=0.3)
            elif plot_type == "Pointplot":
                bar_data = self.df.groupby(y)[x].agg(['mean', 'std']).reset_index()
                fig = px.scatter(bar_data, x=y, y='mean', error_y='std', color=hue, title=title)
            else:
                print("⚠️ Unsupported plot type for Plotly.")
                return
            fig.show()

        if fig:
            self.dashboard.append((fig, title, ftype))

    def bivariate_categorical_categorical(self, cat_cols):
        x = self.choose_from_list("Select first categorical column:", cat_cols)
        y = self.choose_from_list("Select second categorical column:", cat_cols)
        print("\nChoose plot type:")
        plots = ["Countplot", "Grouped Barplot", "Heatmap (Crosstab)"]
        plot_type = self.choose_from_list("Select type:", plots)

        title = f"{plot_type} of {x} vs {y}"
        fig = None
        ftype = "png"
        cross_tab = pd.crosstab(self.df[x], self.df[y])

        if self.backend == "seaborn":
            plt.figure(figsize=(8, 5))
            if plot_type == "Countplot":
                sns.countplot(data=self.df, x=x, hue=y)
            elif plot_type == "Grouped Barplot":
                cross_tab.plot(kind="bar", stacked=False)
                plt.title(title)
            elif plot_type == "Heatmap (Crosstab)":
                sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
            else:
                print("⚠️ Unsupported plot type for Seaborn.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "matplotlib":
            plt.figure(figsize=(8, 5))
            if plot_type == "Countplot":
                counts = self.df.groupby([x, y]).size().unstack(fill_value=0)
                counts.plot(kind="bar", stacked=False)
            elif plot_type == "Grouped Barplot":
                cross_tab.plot(kind="bar", stacked=False)
            elif plot_type == "Heatmap (Crosstab)":
                plt.imshow(cross_tab, cmap="viridis")
                plt.xticks(range(len(cross_tab.columns)), cross_tab.columns, rotation=90)
                plt.yticks(range(len(cross_tab.index)), cross_tab.index)
                plt.colorbar(label="count")
            else:
                print("⚠️ Unsupported plot type for Matplotlib.")
                return
            plt.title(title)
            plt.tight_layout()
            plt.show()
            fig = plt.gcf()

        elif self.backend == "plotly":
            if plot_type == "Countplot":
                fig = px.histogram(self.df, x=x, color=y, barmode="group", title=title)
            elif plot_type == "Grouped Barplot":
                ct_reset = cross_tab.reset_index().melt(id_vars=x, var_name=y, value_name="count")
                fig = px.bar(ct_reset, x=x, y="count", color=y, barmode="group", title=title)
            elif plot_type == "Heatmap (Crosstab)":
                fig = px.imshow(cross_tab.values, x=cross_tab.columns.astype(str), y=cross_tab.index.astype(str), color_continuous_scale="Blues", title=title)
            else:
                print("⚠️ Unsupported plot type for Plotly.")
                return
            fig.show()

        if fig:
            self.dashboard.append((fig, title, ftype))

    # -------------------------
    # Multivariate Analysis
    # -------------------------
    def multivariate_menu(self):
        numeric = self.df.select_dtypes(include='number').columns.tolist()
        cat = self.df.select_dtypes(include='object').columns.tolist()

        print("\n--- Multivariate Analysis ---")
        print("1. Pairplot")
        print("2. Correlation Heatmap")
        print("3. 3D Scatter Plot")
        print("4. Return")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            hue = None
            if cat:
                if input("Add hue? (y/n): ").strip().lower() == "y":
                    hue = self.choose_from_list("Select hue column:", cat)
            sns.pairplot(self.df[numeric + ([hue] if hue else [])], hue=hue)
            plt.suptitle("Pairplot", y=1.02)
            plt.tight_layout()
            plt.show()

        elif choice == "2":
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.df[numeric].corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()

        elif choice == "3":
            if len(numeric) < 3:
                print("⚠️ Need at least three numeric columns.")
                return
            x = self.choose_from_list("Select X-axis:", numeric)
            y = self.choose_from_list("Select Y-axis:", numeric)
            z = self.choose_from_list("Select Z-axis:", numeric)
            color = None
            if cat:
                if input("Color by? (y/n): ").strip().lower() == "y":
                    color = self.choose_from_list("Select color column:", cat)
            title = "3D Scatter Plot"
            if self.backend == "plotly":
                fig = px.scatter_3d(self.df, x=x, y=y, z=z, color=color, title=title)
                fig.show()
            else:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(self.df[x], self.df[y], self.df[z], c="steelblue")
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_zlabel(z)
                plt.title(title)
                plt.tight_layout()
                plt.show()

        else:
            return



    def datetime_menu(self):
        """
        Datetime Analysis Menu with:
         1. Time Series Plot
         2. Histogram of Month/Day/Hour
         3. Resample & Aggregate by Granularity
         4. Rolling Averages
         5. Calendar-Style Heatmap
        """
        print("\n--- Datetime Analysis ---")
        # 1. Detect datetime columns
        dcols = self.df.select_dtypes(include="datetime64[ns]").columns.tolist()
        if not dcols:
            convert = input("No datetime columns found. Convert object columns to datetime? (y/n): ").strip().lower()
            if convert == "y":
                for c in self.df.select_dtypes(include="object"):
                    try:
                        self.df[c] = pd.to_datetime(self.df[c])
                    except:
                        continue
            dcols = self.df.select_dtypes(include="datetime64[ns]").columns.tolist()

        if not dcols:
            print("⚠️ No datetime columns available.")
            return

        # 2. Let user select which datetime column to work with
        dt_col = self.choose_from_list("Select datetime column:", dcols)

        # 3. Present enhanced menu
        while True:
            print(f"\nDatetime column = '{dt_col}'")
            print("1. Time Series Plot (datetime vs numeric)")
            print("2. Histogram of Month / Day of Week / Hour")
            print("3. Resample & Aggregate by Time Granularity")
            print("4. Rolling Average / Moving Window")
            print("5. Calendar-Style Heatmap of Counts")
            print("6. Return to Main Menu")
            choice = input("Enter choice (1-6): ").strip()

            if choice == "1":
                self._datetime_time_series(dt_col)
            elif choice == "2":
                self._datetime_part_histogram(dt_col)
            elif choice == "3":
                self._datetime_resample_aggregate(dt_col)
            elif choice == "4":
                self._datetime_rolling_average(dt_col)
            elif choice == "5":
                self._datetime_calendar_heatmap(dt_col)
            elif choice == "6":
                return
            else:
                print("❌ Invalid choice. Please try again.")

    def _datetime_time_series(self, dt_col):
        """
        Plot a time series of a numeric column against dt_col, with an option to aggregate
        the data by day, week, month, or year (mean or sum).
        """
        # 1) Choose numeric column
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            print("⚠️ No numeric columns available for time series.")
            return
        y_col = self.choose_from_list("Select numeric Y-axis column:", numeric_cols)

        # 2) Prompt for aggregation frequency
        print("\nAggregate data by:")
        print("1. Raw timestamps (no aggregation)")
        print("2. Day")
        print("3. Week")
        print("4. Month")
        print("5. Year")
        freq_choice = input("Enter choice (1-5): ").strip()

        freq_map = {
            "1": None,
            "2": "D",   # daily
            "3": "W",   # weekly
            "4": "M",   # monthly
            "5": "Y",   # yearly
        }
        freq_str = freq_map.get(freq_choice, None)

        # 3) If aggregated (choice 2-5), choose aggregator (mean or sum)
        agg_func = None
        if freq_str is not None:
            print("\nChoose aggregation function:")
            print("1. Mean")
            print("2. Sum")
            agg_choice = input("Enter choice (1 or 2): ").strip()
            agg_func = "mean" if agg_choice == "1" else "sum"

        # 4) Build a title describing the plot
        if freq_str is None:
            title = f"Time Series (raw): {y_col} over {dt_col}"
        else:
            period_map = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Y": "Yearly"}
            title = f"{period_map[freq_str]} {agg_func.capitalize()}: {y_col} over {dt_col}"

        fig = None
        ftype = "png"

        # 5) Sort & prepare the data
        temp = self.df[[dt_col, y_col]].dropna().sort_values(by=dt_col)
        temp = temp.set_index(dt_col)

        if freq_str is None:
            # No resampling—plot raw timestamps
            if self.backend == "plotly":
                fig = px.line(temp.reset_index(), x=dt_col, y=y_col, title=title)
                fig.show()
            else:
                plt.figure(figsize=(10, 4))
                plt.plot(temp.index, temp[y_col], marker="o", linestyle="-", color="steelblue")
                plt.title(title)
                plt.xlabel(dt_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                fig = plt.gcf()
                plt.show()

        else:
            # Perform resample with chosen frequency and aggregator
            try:
                resampled = getattr(temp[y_col].resample(freq_str), agg_func)()
            except Exception as e:
                print(f"❌ Resampling failed: {e}")
                return

            if self.backend == "plotly":
                fig = px.line(resampled.reset_index(), x=dt_col, y=y_col, title=title)
                fig.show()
            else:
                plt.figure(figsize=(10, 4))
                plt.plot(resampled.index, resampled.values, marker="o", linestyle="-", color="teal")
                plt.title(title)
                plt.xlabel(dt_col)
                plt.ylabel(f"{y_col} ({agg_func})")
                plt.tight_layout()
                fig = plt.gcf()
                plt.show()

        # 6) Capture into the dashboard (if fig exists)
        if fig:
            self.dashboard.append((fig, title, ftype))

    def _datetime_part_histogram(self, dt_col):
        """
        Histogram (countplot) of one of {month, day of week, hour} extracted from dt_col.
        """
        print("\nChoose time component:")
        print("1. Month")
        print("2. Day of Week")
        print("3. Hour")
        part = input("Enter choice (1-3): ").strip()

        title = ""
        fig = None
        ftype = "png"

        if part == "1":
            self.df["__month__"] = self.df[dt_col].dt.month
            title = "Count per Month"
            plt.figure(figsize=(8, 4))
            sns.countplot(x="__month__", data=self.df, palette="Blues")
            plt.xlabel("Month")
            plt.title(title)
            fig = plt.gcf()
            plt.tight_layout()
            plt.show()
            self.df.drop(columns="__month__", inplace=True)

        elif part == "2":
            self.df["__dow__"] = self.df[dt_col].dt.day_name()
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            title = "Count per Day of Week"
            plt.figure(figsize=(8, 4))
            sns.countplot(x="__dow__", data=self.df, order=order, palette="magma")
            plt.xlabel("Day of Week")
            plt.title(title)
            plt.xticks(rotation=45)
            fig = plt.gcf()
            plt.tight_layout()
            plt.show()
            self.df.drop(columns="__dow__", inplace=True)

        elif part == "3":
            self.df["__hour__"] = self.df[dt_col].dt.hour
            title = "Count per Hour"
            plt.figure(figsize=(8, 4))
            sns.countplot(x="__hour__", data=self.df, palette="viridis")
            plt.xlabel("Hour of Day")
            plt.title(title)
            fig = plt.gcf()
            plt.tight_layout()
            plt.show()
            self.df.drop(columns="__hour__", inplace=True)

        else:
            print("❌ Invalid choice.")
            return

        if fig:
            self.dashboard.append((fig, title, ftype))

    def _datetime_resample_aggregate(self, dt_col):
        """
        Allow user to resample the DataFrame by a chosen frequency (e.g., daily, weekly, monthly)
        and then plot the aggregated numeric column (mean or sum).
        """
        # Step A: Select numeric column
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            print("⚠️ No numeric columns available for aggregation.")
            return
        y_col = self.choose_from_list("Select numeric column to aggregate:", numeric_cols)

        # Step B: Choose frequency
        print("\nChoose resampling frequency:")
        freqs = ["D (Daily)", "W (Weekly)", "M (Monthly)", "Q (Quarterly)", "Y (Yearly)"]
        freq_choice = self.choose_from_list("Select frequency:", freqs)
        freq_map = {"D (Daily)": "D", "W (Weekly)": "W", "M (Monthly)": "M", "Q (Quarterly)": "Q", "Y (Yearly)": "Y"}
        freq_str = freq_map[freq_choice]

        # Step C: Choose aggregator
        print("\nChoose aggregation function:")
        aggs = ["Mean", "Sum"]
        agg_choice = self.choose_from_list("Select aggregator:", aggs)
        agg_func = "mean" if agg_choice == "Mean" else "sum"

        title = f"{y_col} {agg_choice} Resampled {freq_str}"
        fig = None
        ftype = "png"

        # Perform resample on an index: set dt_col as index first
        temp = self.df.set_index(pd.DatetimeIndex(self.df[dt_col]))
        res = getattr(temp[y_col].resample(freq_str), agg_func)()

        if self.backend == "plotly":
            fig = px.line(res.reset_index(), x=dt_col, y=y_col, title=title)
            fig.show()
        else:
            plt.figure(figsize=(10, 4))
            plt.plot(res.index, res.values, marker="o", linestyle="-", color="teal")
            plt.title(title)
            plt.xlabel(dt_col)
            plt.ylabel(f"{y_col} ({agg_choice})")
            plt.tight_layout()
            fig = plt.gcf()
            plt.show()

        if fig:
            self.dashboard.append((fig, title, ftype))

    def _datetime_rolling_average(self, dt_col):
        """
        Compute and plot a rolling (moving) average of a chosen numeric column over dt_col.
        """
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            print("⚠️ No numeric columns available for rolling average.")
            return
        y_col = self.choose_from_list("Select numeric column for rolling average:", numeric_cols)

        # Step: Ask window size (days)
        while True:
            try:
                window = int(input("Enter rolling window size (in days, e.g. 7): ").strip())
                if window < 1:
                    raise ValueError
                break
            except ValueError:
                print("❌ Please enter a positive integer.")

        title = f"{y_col} {window}-Day Rolling Average"
        fig = None
        ftype = "png"

        # Create a temporary series indexed by datetime, sorted
        temp = self.df[[dt_col, y_col]].dropna().set_index(dt_col).sort_index()
        rolling_series = temp[y_col].rolling(window=f"{window}D").mean()

        if self.backend == "plotly":
            fig = px.line(rolling_series.reset_index(), x=dt_col, y=y_col, title=title)
            fig.show()
        else:
            plt.figure(figsize=(10, 4))
            plt.plot(rolling_series.index, rolling_series.values, color="purple", linewidth=2)
            plt.title(title)
            plt.xlabel(dt_col)
            plt.ylabel(f"{y_col} (rolling {window}d)")
            plt.tight_layout()
            fig = plt.gcf()
            plt.show()

        if fig:
            self.dashboard.append((fig, title, ftype))

    def _datetime_calendar_heatmap(self, dt_col):
        """
        Create a calendar-style heatmap of daily counts (or daily aggregation of a selected numeric column),
        laid out by ISO week vs. weekday.
        """
        # First, prompt the user whether to use record counts or to aggregate a numeric column
        print("\nCalendar-Style Heatmap Options:")
        print("1. Daily record count")
        print("2. Daily sum of a numeric column")
        print("3. Daily average of a numeric column")
        choice = input("Enter choice (1-3): ").strip()

        # Prepare a DataFrame with a 'date-only' column
        temp = self.df.copy()
        temp["__date_only__"] = temp[dt_col].dt.date
        temp["__date_only__"] = pd.to_datetime(temp["__date_only__"])

        # Determine which value to pivot: count, sum, or mean
        if choice == "1":
            # Just count the number of rows per date
            agg_df = temp.groupby("__date_only__").size().reset_index(name="value")
            title = "Calendar Heatmap of Daily Record Counts"

        elif choice in ["2", "3"]:
            # Prompt user to pick a numeric column
            numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                print("⚠️ No numeric columns available. Defaulting to daily record count.")
                agg_df = temp.groupby("__date_only__").size().reset_index(name="value")
                title = "Calendar Heatmap of Daily Record Counts"
            else:
                num_col = self.choose_from_list("Select numeric column:", numeric_cols)
                if choice == "2":
                    # sum per date
                    agg_df = temp.groupby("__date_only__")[num_col].sum().reset_index(name="value")
                    title = f"Calendar Heatmap of Daily Sum: {num_col}"
                else:
                    # mean per date
                    agg_df = temp.groupby("__date_only__")[num_col].mean().reset_index(name="value")
                    title = f"Calendar Heatmap of Daily Average: {num_col}"
        else:
            print("❌ Invalid choice; returning to menu.")
            return

        # Next: compute ISO week and weekday for each date
        agg_df["week"] = agg_df["__date_only__"].dt.isocalendar().week
        agg_df["weekday"] = agg_df["__date_only__"].dt.dayofweek  # 0 = Monday

        # Pivot into matrix: index=week, columns=weekday, values=value
        pivot = agg_df.pivot(index="week", columns="weekday", values="value").fillna(0)

        # Finally, plot the heatmap
        fig = None
        ftype = "png"
        if self.backend in ["seaborn", "matplotlib"]:
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Value"})
            plt.title(title)
            plt.xlabel("Weekday (0=Mon → 6=Sun)")
            plt.ylabel("ISO Week Number")
            plt.tight_layout()
            fig = plt.gcf()
            plt.show()

        elif self.backend == "plotly":
            z = pivot.values
            x_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            y_labels = pivot.index.astype(str)
            fig = px.imshow(
                z,
                x=x_labels,
                y=y_labels,
                color_continuous_scale="YlOrRd",
                labels=dict(x="Weekday", y="ISO Week", color="Value"),
                title=title
            )
            fig.update_xaxes(side="top")
            fig.show()

        # Save to dashboard
        if fig:
            self.dashboard.append((fig, title, ftype))


# In[ ]:




