import java.util.*;
import java.io.*;
public class ExtractMain {

  private static OneDramaFile oneFile( File f )
      throws FileNotFoundException {
    String season = f.getName().substring( 2, 6 );
    OneDramaFile one = new OneDramaFile();
    one.read( f, season );
    return one;
  }

  public static void main( String[] args )
      throws FileNotFoundException {
    File source, output;
    File[] files;
    source = new File( args[0] );
    output = new File( args[1] );
    if ( source.isDirectory() ) {
      files = source.listFiles();
      Arrays.sort( files, new FileComp() );
    }
    else {
      files = new File[ 1 ];
      files[0] = source;
    }
    PrintStream st = new PrintStream( output );
    st.println( DramaObject.header() );
    for ( int j = 0; j < files.length; j ++ ) {
      if ( files[j].getName().endsWith( "html" ) ) {
        System.out.println( files[j].getName() );
        OneDramaFile one = oneFile( files[j] );
        one.encode( st );
      }
    }
    st.flush();
    st.close();
  }
}
