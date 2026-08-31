# app.R
library(shiny)
library(ggplot2)
library(glmnet)
library(dplyr)
library(tidyr)
library(shinythemes)

ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Teaching Linear Model (OLS) and LASSO"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Data Settings"),
      sliderInput("n", "Number of observations (n):", 20, 500, 150, step = 10),
      sliderInput("p", "Number of predictors (p):", 1, 12, 3),
      sliderInput("noise", "Noise (σ):", 0, 3, 1, step = 0.1),
      checkboxInput("sparse_beta", "Make some true coefficients zero (sparse)", TRUE),
      actionButton("gen", "Generate New Data"),
      hr(),
      h4("Model Settings"),
      sliderInput("lambda", "LASSO λ:", min = 0, max = 2, value = 0.2, step = 0.02),
      checkboxInput("standardize", "Standardize predictors (glmnet)", TRUE)
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Overview",
                 h4("Concepts"),
                 p("Compare Ordinary Least Squares (OLS) and LASSO regression."),
                 tags$ul(
                   tags$li("OLS fits by minimizing the sum of squared residuals."),
                   tags$li("LASSO adds an L1 penalty to encourage sparse coefficients:"),
                   withMathJax("$$\\min_\\beta\\; \\|y - X\\beta\\|_2^2 + \\lambda \\sum_{j=1}^p |\\beta_j|$$"),
                   tags$li("As λ increases, LASSO shrinks coefficients; some become exactly zero.")
                 )
        ),
        tabPanel("Data & Model Fit",
                 plotOutput("dataPlot", height = "420px"),
                 tags$hr(),
                 verbatimTextOutput("modelSummary")
        ),
        tabPanel("Coefficient Comparison",
                 plotOutput("coefPlot", height = "420px"),
                 tableOutput("coefTable")
        ),
        tabPanel("LASSO Path",
                 plotOutput("lassoPath", height = "420px"),
                 helpText("Dashed red line marks the current λ on the log(λ) axis.")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  set.seed(123)
  
  # ---- Data generation ----
  data_gen <- eventReactive(input$gen, {
    n <- input$n; p <- input$p; sigma <- input$noise
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    colnames(X) <- paste0("X", 1:p)
    
    # Create a true beta; optionally sparse
    if (input$sparse_beta && p > 2) {
      # Randomly set about half of the coefficients to zero
      nonzero <- sample(1:p, size = ceiling(p/2))
      beta <- rep(0, p)
      beta[nonzero] <- runif(length(nonzero), -3, 3)
    } else {
      beta <- runif(p, -3, 3)
    }
    
    y <- as.numeric(X %*% beta + rnorm(n, sd = sigma))
    dat <- data.frame(y = y, X)
    attr(dat, "true_beta") <- setNames(beta, paste0("X", 1:p))
    dat
  }, ignoreInit = FALSE)
  
  # ---- Fit models ----
  fit_models <- reactive({
    dat <- data_gen()
    X <- as.matrix(dat[, -1, drop = FALSE])
    y <- dat$y
    
    # OLS using all predictors by name
    ols <- lm(y ~ ., data = dat)
    
    # LASSO at the chosen lambda (single λ)
    lam <- input$lambda
    # glmnet expects strictly positive λ; if 0, approximate with tiny positive
    lam_use <- if (lam <= 0) 1e-6 else lam
    lasso <- glmnet(
      X, y, alpha = 1, lambda = lam_use,
      standardize = input$standardize
    )
    
    list(ols = ols, lasso = lasso)
  })
  
  # ---- Tab: Data & Model Fit (only p = 1 is plottable in X-y space) ----
  output$dataPlot <- renderPlot({
    dat <- data_gen()
    p <- ncol(dat) - 1
    models <- fit_models()
    
    if (p == 1) {
      xname <- "X1"
      Xvec <- dat[[xname]]
      xgrid <- seq(min(Xvec), max(Xvec), length.out = 200)
      newdat_ols <- data.frame(X1 = xgrid)
      ols_pred <- predict(models$ols, newdata = newdat_ols)
      
      # For glmnet, newx must be a matrix with matching column name
      newx <- matrix(xgrid, ncol = 1)
      colnames(newx) <- "X1"
      lasso_pred <- as.numeric(predict(models$lasso, newx = newx, s = NULL))
      
      ggplot(dat, aes(x = .data[[xname]], y = y)) +
        geom_point(alpha = 0.7) +
        geom_line(aes(x = xgrid, y = ols_pred), linewidth = 1.1) +
        geom_line(aes(x = xgrid, y = lasso_pred), linetype = "dashed", linewidth = 1.1) +
        labs(
          title = "OLS (solid) vs LASSO (dashed)",
          x = xname, y = "y"
        ) +
        theme_minimal(base_size = 14)
    } else {
      ggplot(data.frame(x = 1, y = 1), aes(x, y)) +
        annotate("text", x = 1, y = 1,
                 label = "Scatter + line plot available only when p = 1",
                 size = 5) +
        theme_void()
    }
  })
  
  output$modelSummary <- renderPrint({
    dat <- data_gen()
    true_beta <- attr(dat, "true_beta")
    models <- fit_models()
    
    cat("True coefficients (no intercept):\n")
    print(round(true_beta, 3))
    
    cat("\nOLS coefficients:\n")
    print(round(coef(models$ols), 3))
    
    cat("\nLASSO coefficients (includes intercept; λ =",
        ifelse(input$lambda <= 0, 1e-6, input$lambda), "):\n")
    print(round(as.vector(coef(models$lasso)), 3))
  })
  
  # ---- Tab: Coefficient Comparison ----
  output$coefPlot <- renderPlot({
    dat <- data_gen()
    p <- ncol(dat) - 1
    pred_names <- paste0("X", 1:p)
    true_beta <- attr(dat, "true_beta")
    
    models <- fit_models()
    ols_coef <- coef(models$ols)[pred_names]
    lasso_coef <- as.numeric(coef(models$lasso))[match(pred_names, rownames(coef(models$lasso)))]
    
    df <- tibble(
      Predictor = factor(pred_names, levels = pred_names),
      True = as.numeric(true_beta[pred_names]),
      OLS = as.numeric(ols_coef),
      LASSO = lasso_coef
    ) |>
      pivot_longer(-Predictor, names_to = "Model", values_to = "Coefficient")
    
    ggplot(df, aes(Predictor, Coefficient, fill = Model)) +
      geom_col(position = "dodge") +
      labs(title = "Coefficient Comparison: True vs OLS vs LASSO") +
      theme_minimal(base_size = 14)
  })
  
  output$coefTable <- renderTable({
    dat <- data_gen()
    p <- ncol(dat) - 1
    pred_names <- paste0("X", 1:p)
    true_beta <- attr(dat, "true_beta")
    models <- fit_models()
    
    ols_coef <- coef(models$ols)[pred_names]
    lasso_coef <- as.numeric(coef(models$lasso))[match(pred_names, rownames(coef(models$lasso)))]
    
    data.frame(
      Predictor = pred_names,
      True = round(as.numeric(true_beta[pred_names]), 3),
      OLS = round(as.numeric(ols_coef), 3),
      LASSO = round(lasso_coef, 3),
      row.names = NULL,
      check.names = FALSE
    )
  })
  
  # ---- Tab: LASSO Path ----
  output$lassoPath <- renderPlot({
    dat <- data_gen()
    X <- as.matrix(dat[, -1, drop = FALSE])
    y <- dat$y
    
    fit <- glmnet(X, y, alpha = 1, standardize = input$standardize) # full path
    plot(fit, xvar = "lambda", label = TRUE)
    
    if (input$lambda > 0) {
      abline(v = log(input$lambda), col = "red", lwd = 2, lty = 2)
    } else {
      # Mark near-zero λ if chosen
      abline(v = log(min(fit$lambda)), col = "red", lwd = 2, lty = 2)
    }
  })
}

shinyApp(ui, server)
