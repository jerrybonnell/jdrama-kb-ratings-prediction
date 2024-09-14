import java.util.*;
import java.io.*;
public class DramaObjectTest {

  /**
   * A class for testing the functionality of DramaObject
   */
  public static void main( String[] args ) throws FileNotFoundException {
    Scanner sc = new Scanner( new File( args[ 0 ] ) );
    ArrayList< String > lines = new ArrayList< String >();
    while ( sc.hasNext() ) { lines.add( sc.nextLine() ); }
    sc.close();
    DramaObject object = new DramaObject( "1004" );
    object.analyze( lines );
    object.print();
    System.out.println( object.encode() );
    PrintStream out =new PrintStream( new File( "test.csv" ) );
    out.println( DramaObject.header() );
    out.println( object.encode() );
    out.flush();
    out.close();
  }

}
