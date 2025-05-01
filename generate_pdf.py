import os
import subprocess
import platform

def generate_pdf_from_markdown(markdown_file, output_pdf):
    """
    Generate a PDF from a Markdown file using mdpdf or pandoc
    """
    try:
        # First try using mdpdf (Node.js based)
        print("Attempting to generate PDF using mdpdf...")
        subprocess.run(["mdpdf", markdown_file, output_pdf], check=True)
        print(f"PDF successfully generated at {output_pdf}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("mdpdf not available or failed. Trying pandoc...")
        
        try:
            # Try using pandoc as an alternative
            subprocess.run([
                "pandoc", 
                markdown_file, 
                "-o", output_pdf,
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=11pt",
                "--toc"
            ], check=True)
            print(f"PDF successfully generated at {output_pdf}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Pandoc not available or failed.")
            
            # If on Windows, try using Microsoft Word as a last resort
            if platform.system() == "Windows":
                try:
                    print("Trying to use Microsoft Word (Windows only)...")
                    # First convert MD to DOCX
                    docx_file = markdown_file.replace(".md", ".docx")
                    subprocess.run([
                        "pandoc", 
                        markdown_file, 
                        "-o", docx_file
                    ], check=True)
                    
                    # Then use Word to convert DOCX to PDF
                    import win32com.client
                    word = win32com.client.Dispatch("Word.Application")
                    doc = word.Documents.Open(os.path.abspath(docx_file))
                    doc.SaveAs(os.path.abspath(output_pdf), FileFormat=17)
                    doc.Close()
                    word.Quit()
                    print(f"PDF successfully generated at {output_pdf}")
                    return True
                except Exception as e:
                    print(f"Microsoft Word conversion failed: {e}")
            
            print("\nCould not generate PDF automatically. Please try one of these methods manually:")
            print("1. Install Node.js and mdpdf: npm install -g mdpdf")
            print("2. Install Pandoc and a LaTeX distribution")
            print("3. Open the markdown file in a markdown editor that supports PDF export")
            print("4. Copy the content to a word processor and save as PDF")
            return False

if __name__ == "__main__":
    markdown_file = "audio_classification_report.md"
    output_pdf = "Audio_Classification_Technical_Report.pdf"
    
    if not os.path.exists(markdown_file):
        print(f"Error: Markdown file {markdown_file} not found.")
    else:
        generate_pdf_from_markdown(markdown_file, output_pdf)
