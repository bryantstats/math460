# app.R
library(shiny)
library(ggplot2)
library(class)
library(dplyr)
library(plotly)

ui <- fluidPage(
  titlePanel("Interactive kNN Classifier Teaching App"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Data and Model Settings"),
      selectInput("xvar", "X Variable:", 
                  choices = names(iris)[1:4], selected = "Sepal.Length"),
      selectInput("yvar", "Y Variable:", 
                  choices = names(iris)[1:4], selected = "Sepal.Width"),
      selectInput("target", "Target Variable (class):", 
                  choices = "Species", selected = "Species"),
      sliderInput("k", "Number of Neighbors (k):", 
                  min = 1, max = 20, value = 5, step = 1),
      selectInput("distance", "Distance Metric:", 
                  choices = c("Euclidean" = 2, "Manhattan" = 1)),
      
      hr(),
      h4("Predict a New Point"),
      numericInput("newx", "New X value:", value = 5.0, step = 0.1),
      numericInput("newy", "New Y value:", value = 3.0, step = 0.1),
      actionButton("add_point", "Add New Point"),
      
      hr(),
      helpText("Adjust k, choose features, and explore how decision boundaries change.
               Add a new point to predict its class.")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Decision Boundary", plotlyOutput("plot_knn", height = "550px")),
        tabPanel("Model Summary", verbatimTextOutput("summary_text")),
        tabPanel("Confusion Matrix", tableOutput("confusion_table"))
      )
    )
  )
)

server <- function(input, output, session) {
  data <- reactive({ iris })
  
  # Reactive KNN prediction (cross-validation)
  pred_knn <- reactive({
    df <- data()
    x <- df[, c(input$xvar, input$yvar)]
    y <- df[[input$target]]
    knn.cv(x, y, k = input$k, prob = TRUE, use.all = TRUE)
  })
  
  # Plot decision boundaries safely
  output$plot_knn <- renderPlotly({
    df <- data()
    xvar <- input$xvar
    yvar <- input$yvar
    target <- input$target
    
    # Create grid safely with unique column names
    xseq <- seq(min(df[[xvar]]) - 0.5, max(df[[xvar]]) + 0.5, length.out = 200)
    yseq <- seq(min(df[[yvar]]) - 0.5, max(df[[yvar]]) + 0.5, length.out = 200)
    grid <- expand.grid(X = xseq, Y = yseq)
    
    # Predict over grid
    pred <- knn(
      train = df[, c(xvar, yvar)],
      test = data.frame(setNames(grid, c(xvar, yvar))),
      cl = df[[target]],
      k = input$k
    )
    grid$pred <- pred
    
    # Base plot
    p <- ggplot(grid, aes(x = X, y = Y, fill = pred)) +
      geom_tile(alpha = 0.3) +
      geom_point(data = df, aes_string(x = xvar, y = yvar, color = target), size = 3) +
      theme_minimal() +
      labs(
        title = paste("kNN Decision Boundary (k =", input$k, ")"),
        fill = "Predicted Class", color = "Actual Class"
      )
    
    # Add new test point (user input)
    if (input$add_point > 0) {
      new_point <- data.frame(x = input$newx, y = input$newy)
      names(new_point) <- c(xvar, yvar)
      pred_new <- knn(
        train = df[, c(xvar, yvar)],
        test = new_point,
        cl = df[[target]],
        k = input$k
      )
      p <- p +
        geom_point(aes(x = input$newx, y = input$newy),
                   color = "black", size = 5, shape = 8) +
        annotate("text", x = input$newx, y = input$newy + 0.2,
                 label = paste("Predicted:", pred_new), color = "black")
    }
    
    ggplotly(p)
  })
  
  # Model summary
  output$summary_text <- renderPrint({
    df <- data()
    preds <- pred_knn()
    accuracy <- mean(preds == df[[input$target]])
    cat("kNN Model Summary\n")
    cat("----------------------------\n")
    cat("Number of observations:", nrow(df), "\n")
    cat("k =", input$k, "\n")
    cat("Distance metric (p):", input$distance, "\n")
    cat("Accuracy (CV):", round(accuracy, 3), "\n")
  })
  
  # Confusion matrix
  output$confusion_table <- renderTable({
    df <- data()
    preds <- pred_knn()
    table(Predicted = preds, Actual = df[[input$target]])
  }, rownames = TRUE)
}

shinyApp(ui, server)
