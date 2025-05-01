@echo off
echo Audio Classification Technical Report - PDF Generator
echo ===================================================
echo.

python generate_pdf.py

echo.
echo If the PDF generation was successful, you should see the file:
echo Audio_Classification_Technical_Report.pdf
echo.
echo If PDF generation failed, you can open the HTML version instead:
echo audio_classification_report.html
echo.
echo Press any key to exit...
pause > nul
