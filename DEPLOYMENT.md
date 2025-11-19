# Deployment Guide

## Railway Deployment

Railway is a modern platform that makes it easy to deploy Python applications. This guide will walk you through deploying the Fleet Optimization Simulator to Railway.

### Prerequisites

1. A [Railway account](https://railway.app/) (free tier available)
2. Your code pushed to a GitHub repository
3. The Railway CLI (optional, but recommended)

### Option 1: Deploy via Railway Dashboard (Easiest)

1. **Sign in to Railway**
   - Go to [railway.app](https://railway.app/)
   - Sign in with your GitHub account

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your fleet optimization simulator repository
   - Railway will automatically detect it's a Python project

3. **Configure Environment**
   - Railway will automatically detect the `Procfile` and `requirements.txt`
   - No additional environment variables are needed for basic deployment

4. **Deploy**
   - Click "Deploy"
   - Railway will:
     - Install dependencies from `requirements.txt`
     - Run the command specified in `Procfile`
     - Assign a public URL to your app

5. **Access Your App**
   - Once deployed, Railway will provide a URL like `https://your-app.railway.app`
   - Click the URL to access your Streamlit app

### Option 2: Deploy via Railway CLI

1. **Install Railway CLI**
   ```bash
   # macOS/Linux
   bash <(curl -fsSL cli.new)

   # Windows (PowerShell)
   iwr https://railway.app/install.ps1 | iex
   ```

2. **Login to Railway**
   ```bash
   railway login
   ```

3. **Initialize Project**
   ```bash
   # From your project directory
   railway init
   ```

4. **Deploy**
   ```bash
   railway up
   ```

5. **Open Your App**
   ```bash
   railway open
   ```

### Option 3: Deploy from GitHub (Automated Deployments)

1. **Push Your Code to GitHub**
   ```bash
   git add .
   git commit -m "Add Railway deployment configuration"
   git push origin main
   ```

2. **Connect to Railway**
   - In Railway dashboard, create a new project
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically deploy on every push to your main branch

### What's Deployed

The deployment includes:

- **Web Interface**: Streamlit UI accessible via the Railway-provided URL
- **All Models**: Both rule-based and greedy optimization models
- **Full Functionality**: All features from the README including:
  - Model selection and comparison
  - Parameter configuration
  - Real-time simulation
  - Scenario comparison
  - Metrics dashboard
  - CSV export

### Configuration Files

The following files enable Railway deployment:

- **`Procfile`**: Tells Railway how to start the application
  ```
  web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
  ```

- **`requirements.txt`**: Lists all Python dependencies
- **`runtime.txt`**: Specifies Python version (3.11.0)

### Resource Usage

**Free Tier:**
- Railway's free tier includes:
  - $5 of usage per month
  - Suitable for development and testing
  - App sleeps after 30 minutes of inactivity

**Hobby/Pro Tier:**
- For production use or extended simulations
- Always-on services
- More compute resources
- Better for running large fleet simulations

### Troubleshooting

**Build Failures:**
- Check the Railway build logs for dependency errors
- Ensure all dependencies in `requirements.txt` are available on PyPI
- Verify Python version compatibility

**App Won't Start:**
- Check Railway logs: `railway logs`
- Verify the `Procfile` command is correct
- Ensure the app runs locally with: `streamlit run app.py`

**Out of Memory:**
- Large simulations (1000+ vehicles, 24+ hours) may need more memory
- Consider upgrading to Railway Pro
- Or reduce simulation parameters in the UI

**Slow Performance:**
- Railway free tier has limited resources
- For better performance, upgrade to Hobby or Pro tier
- Optimize simulation parameters (reduce fleet size, shorter duration)

### Environment Variables (Optional)

You can configure additional settings via Railway environment variables:

```bash
# Set via Railway dashboard or CLI
railway variables set STREAMLIT_SERVER_HEADLESS=true
railway variables set STREAMLIT_SERVER_ENABLE_CORS=false
```

### Custom Domain (Optional)

1. Go to your Railway project settings
2. Navigate to "Domains"
3. Add your custom domain
4. Follow Railway's instructions to configure DNS

### Monitoring and Logs

**View Logs:**
```bash
railway logs
```

**Monitor Resources:**
- View CPU, memory, and network usage in Railway dashboard
- Check deployment history and rollback if needed

### Updating Your Deployment

**Automatic Updates (GitHub connected):**
- Simply push to your main branch
- Railway will automatically rebuild and redeploy

**Manual Updates (CLI):**
```bash
railway up
```

### Cost Estimation

**Typical Usage (Free Tier):**
- Development and testing: Free
- Small simulations: Free
- Personal projects: Free

**Production Usage:**
- Hobby Plan: $5/month (recommended for production)
- Pro Plan: $20/month (for heavy usage)

### Alternative Deployment Options

If Railway doesn't meet your needs, consider:

- **Heroku**: Similar to Railway, with a Procfile-based deployment
- **Streamlit Cloud**: Purpose-built for Streamlit apps (free tier available)
- **Google Cloud Run**: Containerized deployment
- **AWS Elastic Beanstalk**: AWS-based deployment
- **Azure App Service**: Microsoft Azure deployment
- **DigitalOcean App Platform**: Simple cloud deployment

See platform-specific guides for these alternatives.

## Docker Deployment (Alternative)

If you prefer containerized deployment, you can use Docker:

1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run:
   ```bash
   docker build -t fleet-optimizer .
   docker run -p 8501:8501 fleet-optimizer
   ```

3. Deploy to any container platform (Google Cloud Run, AWS ECS, etc.)

## Support

For deployment issues:
- Railway: [Railway Discord](https://discord.gg/railway)
- Streamlit: [Streamlit Community Forum](https://discuss.streamlit.io/)
- This project: Open an issue on GitHub
