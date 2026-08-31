# app.R
library(shiny)
library(ggplot2)
library(glmnet)
library(dplyr)

ui <- fluidPage(
  titlePanel("📘 Learning LASSO Regression"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Simulation Settings"),
      sliderInput("n", "Number of Observations (n):", min = 50, max = 500, value = 100),
      sliderInput("p", "Number of Predictors (p):", min = 2, max = 20, value = 8),
      sliderInput("noise", "Noise Level (σ):", min = 0, max = 5, value = 1, step = 0.1),
      actionButton("simulate", "🔁 Generate Data"),
      hr(),
      h4("LASSO Settings"),
      sliderInput("lambda", "Lambda (Shrinkage Penalty):", 
                  min = -3, max = 2, value = -1, step = 0.1),
      helpText("λ = exp(value) in glmnet scale."),
      checkboxInput("cvfit", "Use Cross-Validation to Choose λ", FALSE),
      actionButton("fit", "⚙️ Fit Model"),
      width = 3
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Overview", 
                 plotOutput("dataPlot"),
                 verbatimTextOutput("dataSummary")),
        
        tabPanel("LASSO Coefficients",
                 plotOutput("coefPlot"),
                 verbatimTextOutput("coefTable")),
        
        tabPanel("Cross-Validation Plot",
                 plotOutput("cvPlot"),
                 helpText("Shows how mean-squared error changes with λ")),
        
        tabPanel("Explanation",
                 h4("What is LASSO?"),
                 p("LASSO (Least Absolute Shrinkage and Selection Operator) adds an L1 penalty 
                   to the regression coefficients. It can shrink some coefficients to zero, 
                   effectively performing variable selection."),
                 h4("Loss Function:"),
                 withMathJax("$$\\text{Minimize } \\frac{1}{2n}||y - X\\beta||_2^2 + \\lambda ||\\beta||_1$$"),
                 h4("Effect of λ:"),
                 tags$ul(
                   tags$li("Small λ → behaves like OLS (many nonzero coefficients)"),
                   tags$li("Large λ → stronger shrinkage, more zeros"),
                   tags$li("Cross-validation helps find optimal λ")
                 ))
      )
    )
  )
)

server <- function(input, output, session) {
  # Reactive dataset
  data_gen <- eventReactive(input$simulate, {
    set.seed(123)
    n <- input$n
    p <- input$p
    X <- matrix(rnorm(n * p), n, p)
    beta <- c(runif(3, 1, 3), rep(0, p - 3))
    y <- X %*% beta + rnorm(n, sd = input$noise)
    list(X = X, y = y, beta = beta)
  }, ignoreNULL = FALSE)
  
  output$dataPlot <- renderPlot({
    df <- data.frame(y = data_gen()$y)
    ggplot(df, aes(x = seq_along(y), y = y)) +
      geom_point(color = "steelblue") +
      geom_smooth(method = "loess", se = FALSE, color = "red") +
      labs(title = "Simulated Response Data", x = "Observation", y = "y")
  })
  
  output$dataSummary <- renderPrint({
    cat("True β coefficients:\n")
    print(round(data_gen()$beta, 3))
  })
  
  # Model fitting
  model_fit <- eventReactive(input$fit, {
    X <- data_gen()$X
    y <- data_gen()$y
    if (input$cvfit) {
      cv.glmnet(X, y)
    } else {
      lambda_val <- exp(input$lambda)
      glmnet(X, y, lambda = lambda_val)
    }
  })
  
  output$coefPlot <- renderPlot({
    fit <- model_fit()
    X <- data_gen()$X
    y <- data_gen()$y
    if (input$cvfit) {
      plot(fit)
      abline(v = log(fit$lambda.min), col = "red", lty = 2)
    } else {
      plot(model_fit(), xvar = "lambda", label = TRUE)
    }
  })
  
  output$cvPlot <- renderPlot({
    if (input$cvfit) plot(model_fit())
  })
  
  output$coefTable <- renderPrint({
    fit <- model_fit()
    if (input$cvfit) {
      coefs <- coef(fit, s = "lambda.min")
      cat("Coefficients at λ_min:\n")
      print(round(as.matrix(coefs), 3))
    } else {
      coefs <- coef(fit)
      cat("Coefficients at λ =", round(exp(input$lambda), 3), "\n")
      print(round(as.matrix(coefs), 3))
    }
  })
}

shinyApp(ui, server)
