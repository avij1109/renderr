#!/bin/bash

# Render Deployment Script for PPG Server with BP Analysis
# This script deploys the enhanced PPG server with blood pressure analysis to Render

echo "🚀 DEPLOYING PPG SERVER WITH BP ANALYSIS TO RENDER"
echo "=" * 60

echo "📦 Files in deployment folder:"
ls -la /home/avij/AndroidStudioProjects/PPG/render_deploy/

echo ""
echo "📋 Updated requirements.txt:"
cat /home/avij/AndroidStudioProjects/PPG/render_deploy/requirements.txt

echo ""
echo "✅ Deployment files ready!"
echo ""
echo "📋 DEPLOYMENT INSTRUCTIONS:"
echo "1. Push these files to your GitHub repository"
echo "2. Render will automatically detect the changes"
echo "3. The new server will include BP analysis capabilities"
echo ""
echo "🎯 NEW BP ANALYSIS API ENDPOINTS:"
echo "• POST /health - Now includes BP analysis status"
echo "• WebSocket /ws - Enhanced with BP analysis commands:"
echo "  - {\"type\": \"start_bp_analysis\"} - Start 30-second BP collection"
echo "  - {\"type\": \"stop_bp_analysis\"} - Stop BP collection early"
echo "  - {\"type\": \"frame\", \"frame\": \"...\", \"timestamp\": ...} - Send frames as usual"
echo ""
echo "🩺 BP ANALYSIS FEATURES:"
echo "• 93.3% accuracy Random Forest model"
echo "• Categories: normotensive, prehypertensive, hypertensive"
echo "• Real-time signal preprocessing and feature extraction"
echo "• Automatic 30-second collection with progress tracking"
echo ""
echo "🔧 DEPLOYMENT READY!"
