library(shiny)
library(ggplot2)
library(dplyr)
library(shinythemes)

ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Gradient Descent for Linear Regression"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Data Settings"),
      sliderInput("n", "Number of points (n):", 20, 300, 100, step = 10),
      sliderInput("noise", "Noise level:", 0, 3, 0.8, step = 0.1),
      actionButton("regen", "Generate New Data"),
      hr(),
      h4("Gradient Descent Settings"),
      numericInput("lr", "Learning rate (η):", 0.05, min = 0.0001, max = 1, step = 0.01),
      sliderInput("iters", "Number of iterations:", 1, 400, 80),
      checkboxInput("show_trail", "Show trajectory lines", TRUE),
      sliderInput("trail_k", "How many trajectory lines:", 3, 20, 8),
      actionButton("runGD", "Run Gradient Descent", class = "btn-primary")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Visualization",
                 plotOutput("plotGD", height = "430px"),
                 hr(),
                 h5("Loss (MSE) over Iterations:"),
                 plotOutput("lossPlot", height = "220px")
        ),
        tabPanel("Concepts",
                 h4("Gradient Descent in Linear Regression"),
                 withMathJax("
          $$\\hat{y} = w_0 + w_1x$$
          $$J(w_0, w_1) = \\frac{1}{2n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2$$
          $$\\text{Updates: } 
            w_0 \\leftarrow w_0 - \\eta \\frac{\\partial J}{\\partial w_0}, \\;
            w_1 \\leftarrow w_1 - \\eta \\frac{\\partial J}{\\partial w_1}$$
          ")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  # Generate data
  data_gen <- eventReactive(input$regen, {
    n <- input$n
    x <- seq(-5, 5, length.out = n)
    y_true <- 2 + 1.5 * x
    y <- y_true + rnorm(n, 0, input$noise)
    tibble(x, y, y_true)
  }, ignoreInit = FALSE)
  
  # Gradient descent function
  gd_fit <- eventReactive(list(input$runGD, input$regen), {
    dat <- data_gen()
    req(nrow(dat) > 0)
    
    x <- dat$x
    y <- dat$y
    n <- length(y)
    lr <- input$lr
    n_iter <- input$iters
    
    # Initialize parameters
    w0 <- runif(1, -1, 1)
    w1 <- runif(1, -1, 1)
    
    loss_hist <- numeric(n_iter)
    coef_hist <- matrix(NA, ncol = 2, nrow = n_iter,
                        dimnames = list(NULL, c("w0","w1")))
    
    for (i in seq_len(n_iter)) {
      y_pred <- w0 + w1 * x
      err <- y_pred - y
      loss_hist[i] <- mean(err^2) / 2
      dw0 <- mean(err)
      dw1 <- mean(err * x)
      w0 <- w0 - lr * dw0
      w1 <- w1 - lr * dw1
      coef_hist[i, ] <- c(w0, w1)
    }
    list(data = dat, coef_hist = coef_hist, loss_hist = loss_hist)
  }, ignoreInit = FALSE)
  
  # Plot: GD progression
  output$plotGD <- renderPlot({
    res <- gd_fit(); req(res)
    dat <- res$data
    H <- as.data.frame(res$coef_hist)
    req(nrow(H) > 0)
    
    p <- ggplot(dat, aes(x, y)) +
      geom_point(alpha = 0.7, color = "gray40") +
      geom_line(aes(y = y_true), color = "green4", linetype = "dashed", linewidth = 1) +
      labs(
        title = "Gradient Descent Learning the Line",
        subtitle = "Green dashed = true line, Blue = learned line",
        x = "x", y = "y"
      ) +
      theme_minimal(base_size = 14)
    
    # optional trajectory
    if (isTRUE(input$show_trail)) {
      k <- max(1, min(input$trail_k, nrow(H)))
      idx <- unique(round(seq(1, nrow(H), length.out = k)))
      trail_df <- lapply(idx, function(i)
        tibble(x = dat$x, y_hat = H$w0[i] + H$w1[i] * dat$x, step = i)
      ) |> bind_rows()
      p <- p +
        geom_line(data = trail_df, aes(y = y_hat, group = step),
                  alpha = 0.3, color = "skyblue4")
    }
    
    # final line
    w_final <- tail(H, 1)
    dat$y_hat_final <- w_final$w0 + w_final$w1 * dat$x
    p + geom_line(aes(y = y_hat_final), color = "blue", linewidth = 1.2)
  })
  
  # Plot: Loss curve
  output$lossPlot <- renderPlot({
    res <- gd_fit(); req(res)
    df_loss <- tibble(iter = seq_along(res$loss_hist), loss = res$loss_hist)
    ggplot(df_loss, aes(iter, loss)) +
      geom_line(color = "darkorange", linewidth = 1) +
      labs(title = "Convergence of Gradient Descent",
           x = "Iteration", y = "Loss (MSE / 2)") +
      theme_minimal(base_size = 13)
  })
}

shinyApp(ui, server)
